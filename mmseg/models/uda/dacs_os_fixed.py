# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix

from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio

from mmseg.core.evaluation.metrics import total_intersect_and_union
import seaborn as sns
from sklearn.metrics import roc_curve
def analyze_max_distribution(gt_target_seg, ema_logits, gt_semantic_seg, src_logit, num_classes, axs, out_dir, local_iter):
    ##* Max Logit ##
    # Find logits that belongs to ID and OOD
    ID_mask = torch.logical_and((gt_target_seg != (num_classes-1)), (gt_target_seg != 255))
    ID_mask = ID_mask.squeeze(1)
    OOD_mask = (gt_target_seg == (num_classes-1))
    OOD_mask = OOD_mask.squeeze(1)
    max_logit = torch.max(ema_logits, dim=1)[0]
    ID_probs = max_logit.detach().cpu()[ID_mask]
    OOD_probs = max_logit.detach().cpu()[OOD_mask]
    src_max_logit = torch.max(src_logit, dim=1)[0].detach().cpu()
    # Mask src_max_logit based on gt_semantic_seg
    src_mask = torch.logical_and((gt_semantic_seg != (num_classes-1)), (gt_semantic_seg != 255))
    src_mask = src_mask.squeeze(1)
    src_max_logit = src_max_logit[src_mask]
    # Calculate for ID vs OOD
    y_true_ood = np.concatenate([np.ones_like(ID_probs), np.zeros_like(OOD_probs)])
    y_scores_ood = np.concatenate([ID_probs, OOD_probs])
    fpr_ood, tpr_ood, thresholds_ood = roc_curve(y_true_ood, y_scores_ood)
    fpr95_ood = fpr_ood[np.where(tpr_ood >= 0.95)[0][0]]
    thres_ood = thresholds_ood[np.where(tpr_ood >= 0.95)[0][0]]

    # Calculate for Source vs OOD
    y_true_src_ood = np.concatenate([np.ones_like(src_max_logit), np.zeros_like(OOD_probs)])
    y_scores_src_ood = np.concatenate([src_max_logit, OOD_probs])
    fpr_src_ood, tpr_src_ood, thresholds_src_ood = roc_curve(y_true_src_ood, y_scores_src_ood)
    fpr95_src_ood = fpr_src_ood[np.where(tpr_src_ood >= 0.95)[0][0]]
    thres_src_ood = thresholds_src_ood[np.where(tpr_src_ood >= 0.95)[0][0]]

    sns.kdeplot(ID_probs, shade=True, label='in-distribution', color='blue', ax=axs)
    sns.kdeplot(OOD_probs, shade=True, label='out-of-distribution', color='gray', ax=axs)
    sns.kdeplot(src_max_logit, shade=True, label='source-distribution', color='green', ax=axs)
    axs.axvline(x=thres_ood, color='red', linestyle='--', linewidth=0.5, label='OOD Threshold')
    axs.axvline(x=thres_src_ood, color='orange', linestyle='--', linewidth=0.5, label='Source Threshold')
    
    # Adding labels and title
    axs.set_title(f'ID - FPR95: {fpr95_ood:.4f}, Threshold: {thres_ood:.4f}\n'
                    f'Source - FPR95: {fpr95_src_ood:.4f}, Threshold: {thres_src_ood:.4f}')
    axs.set_xlabel('Max Logit Score')
    axs.set_ylabel('Frequency')
    axs.legend(loc='upper right')

    plt.savefig(
        os.path.join(out_dir,
                        f'{(local_iter + 1):06d}.png'))
    plt.close()

def analyze_class_distribution(gt_target_seg, ema_logits, gt_semantic_seg, src_logit, num_classes, axs, out_dir, local_iter):
    # Create a figure with subplots for each class
    num_rows = (num_classes - 1 + 2) // 3  # +2 to include OOD
    fig, axs = plt.subplots(num_rows, 3, figsize=(15, 5*num_rows))
    axs = axs.flatten()
    # Prepare data
    max_logit, pred_classes = torch.max(ema_logits, dim=1)
    max_logit = max_logit.detach().cpu()
    pred_classes = pred_classes.detach().cpu()
    src_max_logit, src_pred_classes = torch.max(src_logit, dim=1)
    src_max_logit = src_max_logit.detach().cpu()
    src_pred_classes = src_pred_classes.detach().cpu()
    gt_target_seg = gt_target_seg.squeeze(1).cpu()
    gt_semantic_seg = gt_semantic_seg.squeeze(1).cpu()
    
    # Plot distribution for each class
    for class_idx in range(num_classes - 1):
        ax = axs[class_idx]
        
        # Target domain - ID
        id_mask = (gt_target_seg == class_idx) & (pred_classes == class_idx)
        id_probs = max_logit[id_mask]
        
        if len(id_probs) > 0:
            sns.kdeplot(id_probs, shade=True, label='target ID', color='blue', ax=ax)
        
        # Target domain - OOD
        ood_mask = (gt_target_seg == num_classes-1) & (pred_classes == class_idx)
        ood_probs = max_logit[ood_mask]
        
        if len(ood_probs) > 0:
            sns.kdeplot(ood_probs, shade=True, label='target OOD', color='red', ax=ax)
        
        # Source domain
        src_class_mask = (gt_semantic_seg == class_idx) & (src_pred_classes == class_idx)
        src_class_probs = src_max_logit[src_class_mask]
        
        if len(src_class_probs) > 0:
            sns.kdeplot(src_class_probs, shade=True, label='source', color='green', ax=ax)
        
        ax.set_title(f'Class {class_idx}')
        ax.set_xlabel('Max Logit Score')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    # Plot OOD distribution
    ood_ax = axs[-1]
    ood_mask = (gt_target_seg == (num_classes-1))
    ood_probs = max_logit[ood_mask]
    if len(ood_probs) > 0:
        sns.kdeplot(ood_probs, shade=True, label='OOD', color='red', ax=ood_ax)
    ood_ax.set_title('OOD')
    ood_ax.set_xlabel('Max Logit Score')
    ood_ax.set_ylabel('Frequency')
    ood_ax.legend()
    
    # Remove any unused subplots
    for idx in range(num_classes + 1, len(axs)):
        fig.delaxes(axs[idx])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{(local_iter + 1):06d}_class_dist.png'))
    plt.close()
def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DACS_OS_Fixed(UDADecorator):

    def __init__(self, **cfg):
        super(DACS_OS_Fixed, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold_p = cfg['pseudo_threshold_p'] # unknown threshold
        self.pseudo_threshold_t = cfg['pseudo_threshold_t'] # confidence
        self.unknown_label = cfg['unknown_label'] # the label of the unknown classes
        self.use_raw_logits = cfg['use_raw_logits'] # Use raw logits in pseudo_prob
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

        self.alpha_iou = cfg['alpha_iou']
        self.source_iou = None
        self.source_pretrain = cfg['source_pretrain']
        self.source_selection = cfg['source_selection']
    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas, target_gt):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }
        src_logit = self.get_model().encode_decode(
            img, img_metas)
        _, source_label = torch.max(src_logit, dim=1)
        # Ensure source_label and gt_semantic_seg are in the correct shape and type
        source_label_iou = source_label.squeeze(1).cpu().numpy()  
        gt_semantic_seg_iou = gt_semantic_seg.squeeze(1).cpu().numpy()  
        total_intersect, total_union, _, _ = total_intersect_and_union(
            [source_label_iou],  # Wrap in list as function expects list of arrays
            [gt_semantic_seg_iou],  # Wrap in list as function expects list of arrays
            self.num_classes,
            255
        )
        if self.source_iou is None:
            self.source_iou = (total_intersect / total_union).numpy()
        else:
            # do a EMA update for the source iou
            new_iou = (total_intersect / total_union).numpy()
            mask = ~np.isnan(new_iou) # mask out the ones that are nan in new_iou
            if self.source_selection:
                self.source_iou[mask] = self.alpha_iou * self.source_iou[mask] + \
                    (1 - self.alpha_iou) * new_iou[mask]
        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        ema_logits = self.get_ema_model().encode_decode(
            target_img, target_img_metas)

        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)

        ps_large_t = pseudo_prob.ge(self.pseudo_threshold_t).long() == 1
        if self.use_raw_logits:
            pseudo_prob, pseudo_label = torch.max(ema_logits, dim=1) #! Using raw logits in pseudo_prob
        pseudo_label_copy = pseudo_label.clone().detach() # Copy for further visualization
        pseudo_prob_copy = pseudo_prob.clone().detach() # Copy for further visualization
        #mmcv.print_log(self.local_iter, 'mmseg')
        #ps_small_p = torch.logical_and(ps_small_p, (pseudo_label != 2))
        #pseudo_label[ps_small_p] = self.unknown_label

        if self.unknown_label != None and self.local_iter >= self.source_pretrain: 
            # All pixels with condifence less than pseudo_threshold_p
            ps_small_p = pseudo_prob.lt(self.pseudo_threshold_p).long() == 1 
            if self.source_selection is True:
                for i in range(self.source_iou.shape[0]):
                    if self.source_iou[i] < 0.7:
                        ps_small_p = torch.logical_and(ps_small_p, (pseudo_label != i))
            pseudo_label[ps_small_p] = self.unknown_label
        if self.local_iter % 50 == 0:
            mmcv.print_log(f'source_iou:{self.source_iou}', 'mmseg')

        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_t).item() / ps_size # q_t, confidence
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=dev)

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)
        
        # Train on mixed images
        if self.unknown_label != None and self.local_iter >= self.source_pretrain:
            mix_losses = self.get_model().forward_train(
                mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
            mix_losses.pop('features')
            mix_losses = add_prefix(mix_losses, 'mix')
            mix_loss, mix_log_vars = self._parse_losses(mix_losses)
            log_vars.update(mix_log_vars)
            mix_loss.backward()
        # Analyze max distribution
        if self.local_iter % self.debug_img_interval == 0:
            
            out_dir = os.path.join(self.train_cfg['work_dir'], 'max_logit_debug')
            os.makedirs(out_dir, exist_ok=True)
            
            # Create a new figure for the max logit distribution plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
           
            # Call analyze_class_distribution
            analyze_class_distribution(
                gt_target_seg=target_gt,
                ema_logits=ema_logits,
                gt_semantic_seg=gt_semantic_seg,
                src_logit=src_logit,
                num_classes=self.num_classes,
                axs=ax,
                out_dir=out_dir,
                local_iter=self.local_iter
            )
            
            plt.close(fig)  # Close the figure to free up memory
         
        if self.local_iter % self.debug_img_interval == 0:
            cmap_type = 'isprs'
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap=cmap_type)
                subplotimg(
                    axs[1][1],
                    pseudo_label_copy[j],
                    'Target Pseudo GT',
                    cmap=cmap_type)
                subplotimg(
                    axs[1][2],
                    pseudo_label[j],
                    'Target Threshold (Pseudo) GT',
                    cmap=cmap_type)
                subplotimg(
                    axs[0][2],
                    pseudo_prob_copy[j],
                    'Target probability',
                    cmap='viridis', interpolation='nearest')                
                subplotimg(axs[0][3], vis_mixed_img[j], 'Mixed Image')
                subplotimg(
                    axs[1][3], mix_masks[j][0], 'Domain Mask', cmap='gray')
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                subplotimg(
                    axs[1][4], mixed_lbl[j], 'Seg Targ', cmap=cmap_type)
                subplotimg(
                    axs[0][4], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                """
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        'FDist Mask',
                        cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap=cmap_type)
                """
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
        self.local_iter += 1

        return log_vars
