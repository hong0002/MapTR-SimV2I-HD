_base_ = './simv2i_maptr_ego_r18_compact.py'

# Change this single value to compare 50/75/100-vector baselines.
num_vec = 75
class_balanced_sampling = False
class_balance_oversample_thr = 0.1

model = dict(
    pts_bbox_head=dict(
        num_vec=num_vec,
        bbox_coder=dict(max_num=num_vec),
        transformer=dict(
            encoder=dict(num_layers=2),
            decoder=dict(num_layers=4),
        ),
    )
)

data = dict(samples_per_gpu=4, workers_per_gpu=4)

# SimV2IMapDataset.get_cat_ids supports wrapping train with
# ClassBalancedDataset. Enable only after checking per-class frequencies.
# train = dict(
#     type='ClassBalancedDataset',
#     oversample_thr=class_balance_oversample_thr,
#     dataset=<copy of the compact config train dataset>,
# )

optimizer = dict(
    type='AdamW',
    lr=6e-4,
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
    weight_decay=0.01,
)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
evaluation = dict(interval=2)
checkpoint_config = dict(interval=2, max_keep_ckpts=5)
