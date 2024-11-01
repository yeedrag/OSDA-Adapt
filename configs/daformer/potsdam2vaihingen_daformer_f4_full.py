# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
<<<<<<<< HEAD:configs/daformer/potsdam2vaihingen_daformer_f4_full.py
    '../_base_/models/daformer_sepaspp_bottleneck_mitb5.py',
    # Potsdam->Vaihingen Data Loading
    '../_base_/datasets/uda_potsdam_to_vaihingen_512x512.py',
========
    '../_base_/models/daformer_crf_sepaspp_mitb5_f5.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
>>>>>>>> open_set:configs/daformer/old/gta2cs_tau5_OS_crf_f13.py
    # Basic UDA Self-Training
    '../_base_/uda/dacs.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 0

model = dict(
    decode_head=dict(
        decoder_params=dict(
                # CRF related params
                filter_size=13)))

# Modifications to Basic UDA
uda = dict(
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
    pseudo_threshold_t=0.968, # Original DAFormer Paper
    pseudo_threshold_p=0.5,
    # Pseudo-Label Crop
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120)
data = dict(
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
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
evaluation = dict(interval=2000, metric='mIoU')
# Meta Information for Result Analysis
<<<<<<<< HEAD:configs/daformer/potsdam2vaihingen_daformer_f4_full.py
name = 'potsdam2vaihingen_uda_warm_fdthings_rcs_croppl_a999_daformer_f4'
exp = 'basic'
name_dataset = 'potsdam2vaihingen'
name_architecture = 'daformer_sepaspp_f4_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp_f4'
========
name = 'gta2cs_tau5_OS_crf_f13'
exp = 'basic'
name_dataset = 'gta2cityscapes'
name_architecture = 'daformer_crf_f13_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
>>>>>>>> open_set:configs/daformer/old/gta2cs_tau5_OS_crf_f13.py
name_uda = 'dacs_a999_fd_things_rcs0.01_cpl'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'