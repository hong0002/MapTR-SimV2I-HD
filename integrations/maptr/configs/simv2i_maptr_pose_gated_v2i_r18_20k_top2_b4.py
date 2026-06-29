_base_ = './simv2i_maptr_pose_gated_v2i_r18_20k_b4.py'

work_dir = 'outputs/maptr/pose_gated_v2i_r18_20k_top2_b4'

num_ego_views = 6
num_rsu_views = 2
input_rsu_view_count = 4
num_cams = num_ego_views + num_rsu_views

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)

model = dict(
    ego_view_count=num_ego_views,
    rsu_view_count=num_rsu_views,
    input_rsu_view_count=input_rsu_view_count,
    pose_gate=dict(
        ego_view_count=num_ego_views,
        rsu_view_count=num_rsu_views,
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
