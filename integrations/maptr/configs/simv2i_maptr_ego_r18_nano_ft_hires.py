_base_ = './simv2i_maptr_ego_r18_nano_ft.py'

# Higher input resolution diagnostic. The base SimV2I config scales
# 800x450 images by 0.2 -> 160x90 before padding. This config uses 0.4.
image_scale_factor = 0.4

train_pipeline = [
    dict(type='SimV2ILoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    ),
    dict(type='RandomScaleImageMultiViewImage', scales=[image_scale_factor]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[],
        with_gt=False,
        with_label=False,
    ),
    dict(type='CustomCollect3D', keys=['img']),
]

test_pipeline = [
    dict(type='SimV2ILoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='RandomScaleImageMultiViewImage',
                scales=[image_scale_factor],
            ),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[],
                with_gt=False,
                with_label=False,
            ),
            dict(type='CustomCollect3D', keys=['img']),
        ],
    ),
]

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)
evaluation = dict(pipeline=test_pipeline)

