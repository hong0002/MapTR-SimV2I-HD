import os as _os

_base_ = './simv2i_maptr_v2_dynamic_rsu_top4_r18_20k.py'

dataset_name = 'simv2i_hd_benchmark_v2_dynamic_rsu_20k_geo_subset'
_geo_root = _os.getenv('SIMV2I_HD_GEO_ROOT')
_image_root = _os.getenv('SIMV2I_HD_IMAGE_ROOT')
del _os
dataset_dir = _geo_root or 'data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k_geo_subset'
data_root = _image_root or '.'
eval_cache_dir = 'outputs/maptr_geo/eval_cache'
work_dir = 'outputs/maptr_geo/v2_dynamic_rsu_top4_r18_20k_geo_b4'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=dataset_dir + '/simv2i_maptr_infos_train.pkl',
    ),
    val=dict(
        data_root=data_root,
        ann_file=dataset_dir + '/simv2i_maptr_infos_val.pkl',
        map_ann_file=eval_cache_dir
        + '/simv2i_hd_benchmark_v2_dynamic_rsu_20k_geo_val_gt.json',
    ),
    test=dict(
        data_root=data_root,
        ann_file=dataset_dir + '/simv2i_maptr_infos_test.pkl',
        map_ann_file=eval_cache_dir
        + '/simv2i_hd_benchmark_v2_dynamic_rsu_20k_geo_test_gt.json',
    ),
)
