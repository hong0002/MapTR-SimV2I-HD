import os as _os

_base_ = './simv2i_maptr_pose_gated_v2i_r18_20k_b4.py'

dataset_name = 'simv2i_hd_benchmark_v2_dynamic_rsu_20k'
_simv2i_hd_root = _os.getenv('SIMV2I_HD_ROOT')
del _os
dataset_dir = (
    _simv2i_hd_root
    or 'data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k'
)
eval_cache_dir = 'outputs/maptr/eval_cache'
work_dir = 'outputs/maptr/gate_nopose_v2i_r18_20k_b4'

# When SIMV2I_HD_ROOT is set, both the split PKLs and the image paths stored in
# them are resolved from that external dataset root. Without it, keep the
# original repo-relative data/maptr + data/raw layout for backward compatibility.
data_root = _simv2i_hd_root or '.'

num_ego_views = 6
num_rsu_views = 4
input_rsu_view_count = 4
num_cams = num_ego_views + num_rsu_views
gate_mode = 'nopose'
use_pose_metadata = False

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
        + '/simv2i_hd_benchmark_v2_dynamic_rsu_20k_val_gt.json',
    ),
    test=dict(
        data_root=data_root,
        ann_file=dataset_dir + '/simv2i_maptr_infos_test.pkl',
        map_ann_file=eval_cache_dir
        + '/simv2i_hd_benchmark_v2_dynamic_rsu_20k_test_gt.json',
    ),
)

model = dict(
    ego_view_count=num_ego_views,
    rsu_view_count=num_rsu_views,
    input_rsu_view_count=input_rsu_view_count,
    pose_gate=dict(
        ego_view_count=num_ego_views,
        rsu_view_count=num_rsu_views,
        gate_mode=gate_mode,
        use_pose_metadata=use_pose_metadata,
        nopose_gate_init=0.0,
    ),
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
    ),
)

total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
