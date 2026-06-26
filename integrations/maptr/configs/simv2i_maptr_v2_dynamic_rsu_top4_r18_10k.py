_base_ = './simv2i_maptr_v2i_4rsu_r18_stronger_10k.py'

dataset_name = 'simv2i_hd_benchmark_v2_dynamic_rsu'
dataset_dir = 'data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu'
eval_cache_dir = 'outputs/maptr/eval_cache'
work_dir = 'outputs/maptr/v2_dynamic_rsu_top4_r18_10k'

# Keep the established MapTR metric order. The dynamic-RSU v2 metadata stores
# source labels as divider, ped_crossing, boundary; SimV2IMapDataset maps
# numeric source labels through source class names before applying this order.
map_classes = ['divider', 'boundary', 'ped_crossing']

view_mode = 'v2i_4rsu'
num_cams = 10

# Empty rsu_camera_names means: do not force fixed RSU ids; consume the
# selected top-4 RSU cameras already stored in each sample's rsu_cams field.
dynamic_rsu_camera_names = []

data = dict(
    train=dict(
        ann_file=dataset_dir + '/simv2i_maptr_infos_train.pkl',
        map_classes=map_classes,
        view_mode=view_mode,
        rsu_camera_names=dynamic_rsu_camera_names,
    ),
    val=dict(
        ann_file=dataset_dir + '/simv2i_maptr_infos_val.pkl',
        map_ann_file=eval_cache_dir
        + '/simv2i_hd_benchmark_v2_dynamic_rsu_val_gt.json',
        map_classes=map_classes,
        view_mode=view_mode,
        rsu_camera_names=dynamic_rsu_camera_names,
    ),
    test=dict(
        ann_file=dataset_dir + '/simv2i_maptr_infos_test.pkl',
        map_ann_file=eval_cache_dir
        + '/simv2i_hd_benchmark_v2_dynamic_rsu_test_gt.json',
        map_classes=map_classes,
        view_mode=view_mode,
        rsu_camera_names=dynamic_rsu_camera_names,
    ),
)
