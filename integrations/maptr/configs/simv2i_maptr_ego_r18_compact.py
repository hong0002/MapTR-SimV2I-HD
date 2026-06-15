_base_ = '../../../projects/configs/maptr/maptr_nano_r18_110e.py'

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

dataset_type = 'SimV2IMapDataset'
data_root = '.'
dataset_dir = 'data/maptr/simv2i_hd_v1'

camera_names = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]
map_classes = ['divider', 'boundary', 'ped_crossing']
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

fixed_ptsnum_per_gt_line = 20
fixed_ptsnum_per_pred_line = 20
num_vec = 50
bev_h_ = 80
bev_w_ = 40

model = dict(
    pretrained=None,
    img_backbone=dict(
        norm_cfg=dict(type='BN', requires_grad=True),
    ),
    pts_bbox_head=dict(
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_vec=num_vec,
        num_pts_per_vec=fixed_ptsnum_per_pred_line,
        num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
        num_classes=len(map_classes),
        bbox_coder=dict(
            max_num=num_vec,
            pc_range=point_cloud_range,
            num_classes=len(map_classes),
        ),
        transformer=dict(
            encoder=dict(pc_range=point_cloud_range),
        ),
    ),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            assigner=dict(pc_range=point_cloud_range),
        )
    ),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.2]),
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
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.2]),
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

common_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    camera_names=camera_names,
    map_classes=map_classes,
    modality=input_modality,
    bev_size=(bev_h_, bev_w_),
    pc_range=point_cloud_range,
    fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
    eval_use_same_gt_sample_num_flag=True,
    padding_value=-10000,
    queue_length=1,
    classes=[],
    box_type_3d='LiDAR',
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        **common_dataset,
        ann_file=dataset_dir + '/simv2i_maptr_infos_train.pkl',
        pipeline=train_pipeline,
        test_mode=False,
        filter_empty_gt=True,
    ),
    val=dict(
        **common_dataset,
        ann_file=dataset_dir + '/simv2i_maptr_infos_val.pkl',
        map_ann_file='outputs/maptr/eval_cache/simv2i_hd_v1_val_gt.json',
        pipeline=test_pipeline,
        test_mode=True,
        samples_per_gpu=1,
    ),
    test=dict(
        **common_dataset,
        ann_file=dataset_dir + '/simv2i_maptr_infos_test.pkl',
        map_ann_file='outputs/maptr/eval_cache/simv2i_hd_v1_test_gt.json',
        pipeline=test_pipeline,
        test_mode=True,
    ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

optimizer = dict(
    type='AdamW',
    lr=4e-4,
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
evaluation = dict(interval=1, pipeline=test_pipeline, metric='chamfer')
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
)
