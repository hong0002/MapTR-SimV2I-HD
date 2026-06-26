_base_ = './simv2i_maptr_v2_dynamic_rsu_top4_r18_10k.py'

dataset_name = 'simv2i_hd_benchmark_v2_dynamic_rsu_20k'
dataset_dir = 'data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k'
eval_cache_dir = 'outputs/maptr/eval_cache'
work_dir = 'outputs/maptr/v2_dynamic_rsu_top4_r18_20k'

data = dict(
    train=dict(
        ann_file=dataset_dir + '/simv2i_maptr_infos_train.pkl',
    ),
    val=dict(
        ann_file=dataset_dir + '/simv2i_maptr_infos_val.pkl',
        map_ann_file=eval_cache_dir
        + '/simv2i_hd_benchmark_v2_dynamic_rsu_20k_val_gt.json',
    ),
    test=dict(
        ann_file=dataset_dir + '/simv2i_maptr_infos_test.pkl',
        map_ann_file=eval_cache_dir
        + '/simv2i_hd_benchmark_v2_dynamic_rsu_20k_test_gt.json',
    ),
)
