_base_ = './simv2i_maptr_ego_r18_stronger.py'

dataset_name = 'simv2i_hd_benchmark_v1'
dataset_dir = 'data/maptr/simv2i_hd_benchmark_v1'
eval_cache_dir = 'outputs/maptr/eval_cache'
work_dir = 'outputs/maptr/ego_r18_stronger_10k'

# Keep the established SimV2I baseline class order. The 10k metadata stores
# labels as divider, ped_crossing, boundary; SimV2IMapDataset maps numeric
# source labels through source class names before applying this order.
map_classes = ['divider', 'boundary', 'ped_crossing']

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        ann_file=dataset_dir + '/simv2i_maptr_infos_train.pkl',
        map_classes=map_classes,
    ),
    val=dict(
        ann_file=dataset_dir + '/simv2i_maptr_infos_val.pkl',
        map_ann_file=eval_cache_dir + '/simv2i_hd_benchmark_v1_val_gt.json',
        map_classes=map_classes,
    ),
    test=dict(
        ann_file=dataset_dir + '/simv2i_maptr_infos_test.pkl',
        map_ann_file=eval_cache_dir + '/simv2i_hd_benchmark_v1_test_gt.json',
        map_classes=map_classes,
    ),
)
