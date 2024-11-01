# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/daformer_sepaspp_mitb5.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/uda_pot2vai_veg_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/dacs_os_fixed.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 0
# Modifications to Basic UDA
uda = dict(
    type='DACS_OS_Fixed',
    # Increased Alpha
    alpha=0.999,
	alpha_iou=0.97,
    # Thing-Class Feature Distance
    #imnet_feature_dist_lambda=0.005,
    #imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    #imnet_feature_dist_scale_min_ratio=0.75,
    # Debugging
    debug_img_interval=50,
    # Unknown label
    unknown_label=6,
    # Pseudo-label tau
    pseudo_threshold_t=0.968, # Original DAFormer Paper
    use_raw_logits=True,
    pseudo_threshold_p=2, #No thresholding 
	source_selection=False,
    # Source Pretrain:
	source_pretrain=0,
    # Pseudo-Label Crop
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120)
data = dict(
    samples_per_gpu=4, # changed from 2 to 4
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))
model = dict(
    decode_head=dict(
        num_classes=7,
    ))
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=4000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=1)
evaluation = dict(interval=500, metric='h_score', save_best='h_score') 
# resume_from = "work_dirs/local-basic/241007_2121_potsdam2vaihingen_OS_Source_only_64531/best_h_score_iter_500.pth"
# Meta Information for Result Analysis
name = "pretrain0_pl_2_source_iou_0.7_veg"
#name = 'potsdam2vaihingen_OS_pretrain500_pl_1.5_source_eval'
exp = 'basic'
name_dataset = 'potsdam2vaihingen'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
