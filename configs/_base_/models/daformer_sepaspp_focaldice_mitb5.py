# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# DAFormer (with context-aware feature fusion) with focal + dice loss

_base_ = ['daformer_sepaspp_mitb5.py']

model = dict(
    decode_head=dict(num_classes=6,
                    loss_decode=[dict(type='DiceLoss', use_sigmoid=False, loss_name='loss_dice', loss_weight=1.0), 
                    dict(type='FocalLoss', loss_name='loss_focal', loss_weight=4.0)]))
