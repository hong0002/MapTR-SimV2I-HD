_base_ = './simv2i_maptr_ego_r18_stronger.py'

dataset_name = 'simv2i_hd_benchmark_v2_dynamic_rsu_20k'
dataset_dir = 'data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k'
eval_cache_dir = 'outputs/maptr/eval_cache'
work_dir = 'outputs/maptr/ego_only_r18_20k'

# Keep data_root at repo root because the pkl stores image/lidar paths as
# repo-relative paths such as data/raw/... . The MapTR pkl root is dataset_dir.
data_root = '.'

map_classes = ['divider', 'boundary', 'ped_crossing']
view_mode = 'ego_only'
num_cams = 6
ego_camera_names = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=dataset_dir + '/simv2i_maptr_infos_train.pkl',
        map_classes=map_classes,
        camera_names=ego_camera_names,
        view_mode=view_mode,
    ),
    val=dict(
        data_root=data_root,
        ann_file=dataset_dir + '/simv2i_maptr_infos_val.pkl',
        map_ann_file=eval_cache_dir
        + '/simv2i_hd_benchmark_v2_dynamic_rsu_20k_val_gt.json',
        map_classes=map_classes,
        camera_names=ego_camera_names,
        view_mode=view_mode,
    ),
    test=dict(
        data_root=data_root,
        ann_file=dataset_dir + '/simv2i_maptr_infos_test.pkl',
        map_ann_file=eval_cache_dir
        + '/simv2i_hd_benchmark_v2_dynamic_rsu_20k_test_gt.json',
        map_classes=map_classes,
        camera_names=ego_camera_names,
        view_mode=view_mode,
    ),
)

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            num_cams=num_cams,
            encoder=dict(
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=256,
                            num_levels=1,
                        ),
                        dict(
                            type='GeometrySptialCrossAttention',
                            num_cams=num_cams,
                            pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                            attention=dict(
                                type='GeometryKernelAttention',
                                embed_dims=256,
                                num_heads=4,
                                dilation=1,
                                kernel_size=(3, 5),
                                num_levels=1,
                                im2col_step=192,
                            ),
                            embed_dims=256,
                        ),
                    ]
                )
            ),
        )
    )
)
