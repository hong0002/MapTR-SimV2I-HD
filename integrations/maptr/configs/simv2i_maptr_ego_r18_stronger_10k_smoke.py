_base_ = './simv2i_maptr_ego_r18_stronger_10k.py'

work_dir = 'outputs/maptr/ego_r18_stronger_10k_smoke'

total_epochs = 1
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
