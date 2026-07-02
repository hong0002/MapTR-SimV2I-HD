# SimV2I-HD MapTR Data Requirements

## Copy To The Server

Required:

- `data/maptr/simv2i_hd_v1/simv2i_maptr_infos_train.pkl`
- `data/maptr/simv2i_hd_v1/simv2i_maptr_infos_val.pkl`
- `data/maptr/simv2i_hd_v1/simv2i_maptr_infos_test.pkl`
- Every image referenced by the six ego-camera `data_path` values
- Optional LiDAR files only when a pickle contains `lidar_path`
- Raw run metadata when it is needed for later audit or V2I synchronization

The expected raw tree is:

```text
data/raw/<run_id>/
├── sensors/
│   ├── ego_rgb_front/
│   ├── ego_rgb_front_left/
│   ├── ego_rgb_front_right/
│   ├── ego_rgb_back/
│   ├── ego_rgb_back_left/
│   ├── ego_rgb_back_right/
│   ├── rsu_00_rgb/
│   ├── rsu_01_rgb/
│   ├── rsu_02_rgb/
│   └── rsu_03_rgb/
├── frames.jsonl
├── metadata/
└── annotations/
```

The ego-only baseline does not read RSU images, LiDAR point contents,
CARLA binaries, Windows virtual environments, intermediate visualization
files, or obsolete GT exports.

## Pickle Contract

- Pickle protocol must be `4` or lower for Python 3.7 compatibility.
- Top level should be `{"infos": [...], "metadata": {...}}`; a direct list of
  infos is also accepted.
- `metadata.map_classes` should be
  `["divider", "boundary", "ped_crossing"]`.
- Every info needs `cams` with these keys:
  `CAM_FRONT`, `CAM_FRONT_RIGHT`, `CAM_FRONT_LEFT`, `CAM_BACK`,
  `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`.
- Every selected camera needs `data_path`, `cam_intrinsic` or `intrinsics`,
  and usable calibration.
- Preferred calibration is direct `lidar2img`/`ego2img`. The adapter also
  accepts the existing MapTR `sensor2lidar_rotation` and
  `sensor2lidar_translation` fields.

All paths should be relative to the configured dataset root. In the legacy
workspace fallback, this was the MapTR project root:

```text
data/raw/town10_maptr_v1/sensors/ego_rgb_front/15971219.png
```

Forbidden:

- Windows absolute paths
- Backslash-separated Windows paths
- Any path containing `_invalid_pre_od_yflip`
- GT generated before the OpenDRIVE y-axis correction

The corrected transform is:

```text
carla_x   = od_x
carla_y   = -od_y
carla_yaw = -od_heading
```

Map vectors must already be in the ego frame:

- x is right-positive, range `[-15, 15]`
- y is forward-positive, range `[-30, 30]`

## Supported GT Layouts

Preferred:

```python
info["gt_vectors"] = [
    {"pts": [[x0, y0], [x1, y1], ...], "cls_name": "divider"},
]
```

Also accepted:

```python
info["map_annos"] = {
    "divider": [polyline0, polyline1],
    "boundary": [polyline2],
    "ped_crossing": [polyline3],
}
```

Or paired arrays:

```python
info["gt_vecs"] = [polyline0, polyline1]
info["gt_labels"] = [0, 1]
```

Numeric labels use the declared `map_classes` order. Each vector needs at
least two finite 2D points.

## Validate After Transfer

```bash
export MAPTR_ROOT=/path/to/MapTR
cd "${MAPTR_ROOT}"
bash scripts/validate_server_maptr_data.sh --require-data
```

Reports are written to:

- `outputs/maptr/server_setup/data_integrity_report.json`
- `outputs/maptr/server_setup/data_integrity_report.md`
