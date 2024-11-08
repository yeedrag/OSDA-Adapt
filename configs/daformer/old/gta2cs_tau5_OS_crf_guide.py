# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/daformer_crf_sepaspp_mitb5.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
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
    type='DACS_OS_Dynamic_Guide',
    # Increased Alpha
    alpha=0.999,
    # Thing-Class Feature Distance
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75,
    # Debugging
    debug_img_interval=500,
    # Unknown label
    unknown_label=19,
    # Pseudo-label tau
    update_threshold_iter=1000,
    pseudo_threshold_t=0.968, # Original DAFormer Paper
    pseudo_threshold_p=0.5,
    # Pseudo-Label Crop
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120)
data = dict(
    samples_per_gpu=4, # changed from 2 to 4
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))
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
runner = dict(type='IterBasedRunner', max_iters=20000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=20000, max_keep_ckpts=1)
evaluation = dict(interval=1000, metric='mIoU')
# Meta Information for Result Analysis
name = 'gta2cs_tau5_OS_crf_guide'
exp = 'basic'
name_dataset = 'gta2cityscapes'
name_architecture = 'daformer_crf_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs_a999_fd_things_rcs0.01_cpl_dynamic'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
