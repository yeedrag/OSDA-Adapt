import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import os

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