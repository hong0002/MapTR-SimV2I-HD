_base_ = './simv2i_maptr_ego_r18_stronger.py'

# Official MapTR-nano R18+GKT checkpoint:
# https://drive.google.com/file/d/1-wVO1pZhFif2igJoz-s451swQvPSto2m/view
load_from = 'ckpts/maptr_nano_r18_110e_official.pth'
resume_from = None

# Official nano uses 100 vectors. Keep the stronger encoder/decoder depth from
# simv2i_maptr_ego_r18_stronger.py, but avoid shape mismatch on query weights.
# Keep the inherited SimV2I class order: ['divider', 'boundary', 'ped_crossing'].
num_vec = 100

model = dict(
    pretrained=None,
    pts_bbox_head=dict(
        num_vec=num_vec,
        bbox_coder=dict(max_num=num_vec),
    ),
)

checkpoint_config = dict(interval=2, max_keep_ckpts=5)
evaluation = dict(interval=2)
