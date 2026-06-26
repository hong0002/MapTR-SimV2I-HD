_base_ = './simv2i_maptr_ego_only_r18_20k_b4.py'

work_dir = 'outputs/maptr/ego_only_r18_20k_b4_smoke'

# 14000/200 = 70 train samples, 2000/200 = 10 val samples.
smoke_load_interval = 200

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(load_interval=smoke_load_interval),
    val=dict(
        load_interval=smoke_load_interval,
        samples_per_gpu=1,
        map_ann_file='outputs/maptr/eval_cache/'
        'simv2i_hd_benchmark_v2_dynamic_rsu_20k_ego_only_b4_smoke_val_gt.json',
    ),
    test=dict(
        load_interval=smoke_load_interval,
        map_ann_file='outputs/maptr/eval_cache/'
        'simv2i_hd_benchmark_v2_dynamic_rsu_20k_ego_only_b4_smoke_test_gt.json',
    ),
)

total_epochs = 1
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
)
