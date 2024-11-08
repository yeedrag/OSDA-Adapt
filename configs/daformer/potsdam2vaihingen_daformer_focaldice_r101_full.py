# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
# Modified for the task potsdam->vaihingen
# This is for using the full mask! Use the switch first!
_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/segformer_r101.py',
    # Potsdam->Vaihingen Data Loading
    '../_base_/datasets/uda_potsdam_to_vaihingen_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/dacs.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]

# Random Seed
seed = 0

# Modifications to Basic UDA
uda = dict(
    # Increased Alpha
    alpha=0.999,
    # Thing-Class Feature Distance
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],   
    imnet_feature_dist_scale_min_ratio=0.75,
    # Pseudo-Label Crop
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120
)

data = dict(
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5
        )
    )
)

odel = dict(
    decode_head=dict(num_classes=6,
                    loss_decode=[dict(type='DiceLoss', use_sigmoid=False, loss_name='loss_dice', loss_weight=1.0), 
                    dict(type='FocalLoss', loss_name='loss_focal', loss_weight=4.0)]))

# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0)
        )
    )
)

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)

# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
evaluation = dict(interval=2000, metric='mIoU')

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
])

# Meta Information for Result Analysis
name = 'potsdam2vaihingen_uda_warm_fdthings_rcs_croppl_a999_daformer_3focal1dice_r101_s0'
exp = 'basic'
name_dataset = 'potsdam2vaihingen'
name_architecture = 'daformer_sepaspp_focaldice_r101'
name_encoder = 'r101'
name_decoder = 'daformer_sepaspp_focaldice'
name_uda = 'dacs_a999_fd_things_rcs0.01_cpl'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'

