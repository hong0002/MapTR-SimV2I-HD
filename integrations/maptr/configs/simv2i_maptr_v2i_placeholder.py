# Design-only placeholder. This file intentionally defines no trainable model.
placeholder_only = True
implementation_status = 'TODO: separate ego/RSU encoders and BEV fusion'

ego_camera_names = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]
rsu_camera_names = [
    'RSU_00_CAM',
    'RSU_01_CAM',
    'RSU_02_CAM',
    'RSU_03_CAM',
]

camera_modes = dict(
    ego_only=ego_camera_names,
    rsu_only=rsu_camera_names,
    # Useful only as an input/calibration diagnostic, not the proper V2I model.
    naive_flattened=ego_camera_names + rsu_camera_names,
)

proper_v2i_design = dict(
    ego_branch='shared or dedicated image backbone + ego-frame BEV encoder',
    infrastructure_branch='RSU image backbone + RSU-frame BEV encoder',
    alignment='timestamp matching and calibrated RSU-BEV to ego-BEV transform',
    fusion='masked attention or confidence-aware BEV fusion',
    output_head='MapTR vector head in the ego coordinate frame',
    required_metadata=[
        'camera timestamp',
        'camera intrinsics/extrinsics',
        'RSU pose in world',
        'ego pose in world',
        'valid-view mask',
    ],
)
