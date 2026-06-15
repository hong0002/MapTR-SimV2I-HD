# MapTR Server Setup

## Recorded State

- SimV2I-HD integration root: `/data1/jihong/MapTR`
- MapTR repo root: `/data1/jihong/MapTR`
- Clone layout: direct MapTR clone, no `third_party/MapTR` wrapper
- Remote: `https://github.com/hustvl/MapTR.git`
- Commit: `a6872d8d9670bde17b4b01560f1221f88b443d55`
- Initial Git state: clean, `main...origin/main`
- MapTR configs: `projects/configs/maptr`
- SimV2I configs: `integrations/maptr/configs`
- Train entrypoint: `tools/train.py`
- Test entrypoint: `tools/test.py`
- Bundled MMDetection3D: `mmdetection3d`, version `0.17.2`

The full machine-readable snapshot is
`outputs/maptr/server_setup/env_summary.json`.

## Current Environment

The active base environment is not a usable MapTR environment:

- Python `3.12.4`
- imported PyTorch `2.11.0+cu130`
- `mmcv`, `mmdet`, and installed `mmdet3d`: absent
- PyTorch CUDA availability: false
- CUDA toolkit: `12.4`
- PyTorch package metadata is inconsistent: both `2.11.0` and `2.6.0`
  metadata directories exist.

The first GPU probe showed four RTX 3090 GPUs with driver `535.183.01`.
Later probes failed to communicate with the NVIDIA driver. Do not begin
training until both commands are stable:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## Recommended Conda Environment

The checked-in MapTR code is a legacy OpenMMLab 1.x stack. Its own guards
require MMDetection3D `0.17.2`, MMCV `1.3.8` through `1.4.0`,
MMDetection `2.14.0` through `<3.0.0`, and MMSegmentation `0.14.1`
through `<1.0.0`.

Use the reference environment as a starting point:

```bash
cd /data1/jihong/MapTR
conda env create -f integrations/maptr/environment/maptr_legacy.yml
conda activate maptr_simv2i
```

For `mmcv-full`, use the CUDA 11.1/PyTorch 1.9 wheel index if plain pip
tries to compile it:

```bash
pip install mmcv-full==1.4.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.14.0 mmsegmentation==0.14.1
```

Install the bundled source and MapTR CUDA operation:

```bash
cd /data1/jihong/MapTR/mmdetection3d
python setup.py develop

cd /data1/jihong/MapTR/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
python setup.py build install
```

Then confirm:

```bash
cd /data1/jihong/MapTR
PYTHONPATH="$PWD:$PWD/mmdetection3d" python -c \
  "import torch, mmcv, mmdet, mmdet3d; print(torch.__version__, mmcv.__version__, mmdet.__version__, mmdet3d.__version__, torch.cuda.is_available())"
```

## SimV2I Integration

- Dataset adapter:
  `projects/mmdet3d_plugin/datasets/simv2i_map_dataset.py`
- Compact smoke config:
  `integrations/maptr/configs/simv2i_maptr_ego_r18_compact.py`
- Stronger ego config:
  `integrations/maptr/configs/simv2i_maptr_ego_r18_stronger.py`
- V2I design placeholder:
  `integrations/maptr/configs/simv2i_maptr_v2i_placeholder.py`

The adapter selects only the six vehicle cameras, consumes precomputed local
vector GT, and does not load point clouds. Coordinates are x-right,
y-forward with x in `[-15, 15]` and y in `[-30, 30]`.

Supported GT layouts are documented in
`docs/MAPTR_DATA_REQUIREMENTS.md`.

## Data Placement

Place the pickles here:

```text
data/maptr/simv2i_hd_v1/
├── simv2i_maptr_infos_train.pkl
├── simv2i_maptr_infos_val.pkl
└── simv2i_maptr_infos_test.pkl
```

Referenced images remain under project-relative paths such as:

```text
data/raw/town10_maptr_v1/sensors/ego_rgb_front/15971219.png
```

## Validation And Runs

Validation is safe before data transfer and exits successfully with
`missing_data`:

```bash
bash scripts/validate_server_maptr_data.sh
```

After transfer, require complete data:

```bash
bash scripts/validate_server_maptr_data.sh --require-data
```

Run the two-epoch camera-only smoke experiment:

```bash
conda activate maptr_simv2i
bash scripts/run_maptr_ego_smoke.sh
```

Run the stronger 24-epoch baseline:

```bash
MAPTR_GPUS=4 bash scripts/run_maptr_ego_train.sh
```

Evaluate a checkpoint:

```bash
bash scripts/eval_maptr_ego.sh \
  outputs/maptr/ego_r18_stronger/latest.pth
```

Each run validates the data first and writes logs below its work directory.
No training or evaluation was run during this setup because the dataset is
not present and the current Python/CUDA environment is not ready.

## V2I Status

The current executable configs are ego-only. The V2I placeholder separates a
naive ten-camera concatenation diagnostic from the intended two-branch,
calibration-aware BEV fusion design. See `docs/MAPTR_V2I_PLAN.md`.
