_base_ = './simv2i_maptr_pose_gated_v2i_r18_20k_b4.py'

dataset_name = 'simv2i_hd_benchmark_v2_dynamic_rsu_20k'
dataset_dir = 'data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k'
eval_cache_dir = 'outputs/maptr/eval_cache'
work_dir = 'outputs/maptr/gate_constant_v2i_r18_20k_b4'

# The pkl stores image paths as repo-relative data/raw/... paths, so the
# runtime data_root must stay at repo root. dataset_dir is the MapTR pkl root.
data_root = '.'

num_ego_views = 6
num_rsu_views = 4
input_rsu_view_count = 4
num_cams = num_ego_views + num_rsu_views
gate_mode = 'constant'
constant_gate_value = 0.2
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
        constant_gate_value=constant_gate_value,
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
