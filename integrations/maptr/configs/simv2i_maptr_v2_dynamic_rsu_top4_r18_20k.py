import os as _os

_base_ = './simv2i_maptr_v2_dynamic_rsu_top4_r18_10k.py'

dataset_name = 'simv2i_hd_benchmark_v2_dynamic_rsu_20k'
_simv2i_hd_root = _os.getenv('SIMV2I_HD_ROOT')
del _os
dataset_dir = (
    _simv2i_hd_root
    or 'data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k'
)
data_root = _simv2i_hd_root or '.'
eval_cache_dir = 'outputs/maptr/eval_cache'
work_dir = 'outputs/maptr/v2_dynamic_rsu_top4_r18_20k'

data = dict(
    train=dict(
        data_root=data_root,
        ann_file=dataset_dir + '/simv2i_maptr_infos_train.pkl',
    ),
    val=dict(
        data_root=data_root,
        ann_file=dataset_dir + '/simv2i_maptr_infos_val.pkl',
        map_ann_file=eval_cache_dir
        + '/simv2i_hd_benchmark_v2_dynamic_rsu_20k_val_gt.json',
    ),
    test=dict(
        data_root=data_root,
        ann_file=dataset_dir + '/simv2i_maptr_infos_test.pkl',
        map_ann_file=eval_cache_dir
        + '/simv2i_hd_benchmark_v2_dynamic_rsu_20k_test_gt.json',
    ),
)
