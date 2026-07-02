import os as _os

_base_ = './simv2i_maptr_v2_dynamic_rsu_top4_r18_20k.py'

dataset_name = 'simv2i_hd_benchmark_v2_dynamic_rsu_20k'
_simv2i_hd_root = _os.getenv('SIMV2I_HD_ROOT')
del _os
dataset_dir = (
    _simv2i_hd_root
    or 'data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k'
)
eval_cache_dir = 'outputs/maptr/eval_cache'
work_dir = 'outputs/maptr/v2_dynamic_rsu_top4_r18_20k_b16'

# When SIMV2I_HD_ROOT is set, both the split PKLs and the image paths stored in
# them are resolved from that external dataset root. Without it, keep the
# original repo-relative data/maptr + data/raw layout for backward compatibility.
data_root = _simv2i_hd_root or '.'

map_classes = ['divider', 'boundary', 'ped_crossing']
view_mode = 'v2i_4rsu'
num_cams = 10
dynamic_rsu_camera_names = []

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=dataset_dir + '/simv2i_maptr_infos_train.pkl',
        map_classes=map_classes,
        view_mode=view_mode,
        rsu_camera_names=dynamic_rsu_camera_names,
    ),
    val=dict(
        data_root=data_root,
        ann_file=dataset_dir + '/simv2i_maptr_infos_val.pkl',
        map_ann_file=eval_cache_dir
        + '/simv2i_hd_benchmark_v2_dynamic_rsu_20k_val_gt.json',
        map_classes=map_classes,
        view_mode=view_mode,
        rsu_camera_names=dynamic_rsu_camera_names,
    ),
    test=dict(
        data_root=data_root,
        ann_file=dataset_dir + '/simv2i_maptr_infos_test.pkl',
        map_ann_file=eval_cache_dir
        + '/simv2i_hd_benchmark_v2_dynamic_rsu_20k_test_gt.json',
        map_classes=map_classes,
        view_mode=view_mode,
        rsu_camera_names=dynamic_rsu_camera_names,
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
