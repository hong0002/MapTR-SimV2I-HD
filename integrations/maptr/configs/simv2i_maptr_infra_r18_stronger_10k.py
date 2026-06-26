_base_ = './simv2i_maptr_ego_r18_stronger_10k.py'

view_mode = 'infra_only_4rsu'
num_cams = 4
work_dir = 'outputs/maptr/infra_r18_stronger_10k'

data = dict(
    train=dict(view_mode=view_mode),
    val=dict(view_mode=view_mode),
    test=dict(view_mode=view_mode),
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
                            num_levels=1),
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
                                im2col_step=192),
                            embed_dims=256)
                    ]
                )
            )
        )
    )
)
