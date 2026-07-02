# SimV2I-HD MapTR

This repository contains the MapTR-based code used for SimV2I-HD HD map
construction experiments. It keeps the original MapTR/MMDetection3D stack and
adds SimV2I-HD dataset adapters, V2I camera configurations, pose-gated RSU
fusion experiments, and paper-oriented analysis/visualization utilities.

The public-facing goal of this repository is reproducibility: keep the paper
experiments runnable from documented configs and scripts without committing
large datasets, checkpoints, prediction files, or generated result folders.

## What Is Included

- SimV2I-HD MapTR configs under `integrations/maptr/configs/`.
- Ego-only, Dynamic Top-4 RSU, and Pose-gated Top-4 V2I training/evaluation
  scripts under `scripts/`.
- Dataset validation tooling under `integrations/maptr/tools/`.
- Qualitative visualization and gate-analysis tools under `tools/`.
- The vendored legacy MMDetection3D stack required by MapTR under
  `mmdetection3d/`.

Large local artifacts are intentionally ignored:

- `data/`
- `ckpts/`
- `outputs/`
- experiment logs, TensorBoard logs, checkpoints, and prediction PKLs

## Repository Layout

```text
integrations/maptr/configs/     Paper experiment configs
integrations/maptr/environment/ Legacy conda environment file
integrations/maptr/tools/       Dataset validation utilities
projects/mmdet3d_plugin/        MapTR model, dataset, and fusion modules
scripts/                        Public wrappers and experiment scripts
tools/                          Training, testing, visualization, analysis tools
docs/                           Additional setup and data notes
```

## Quick Start

Create the environment and build the legacy OpenMMLab extensions:

```bash
export MAPTR_ROOT=/path/to/MapTR-SimV2I-HD
export SIMV2I_HD_ROOT=/path/to/simv2i_hd_benchmark_v2_dynamic_rsu_20k
cd "${MAPTR_ROOT}"

conda env create -f integrations/maptr/environment/maptr_legacy.yml
conda activate maptr_simv2i

cd "${MAPTR_ROOT}/mmdetection3d"
python setup.py develop

cd "${MAPTR_ROOT}/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn"
python setup.py build install
```

For detailed setup notes, see [INSTALL.md](INSTALL.md).

## Data

The full SimV2I-HD dataset is not included in this repository. Set
`SIMV2I_HD_ROOT` to the prepared MapTR-format SimV2I-HD dataset directory. The
directory should contain the train/val/test annotation PKLs, vectorized map GT
JSON files, and image folders referenced by the PKLs:

```text
${SIMV2I_HD_ROOT}/
├── simv2i_maptr_infos_train.pkl
├── simv2i_maptr_infos_val.pkl
├── simv2i_maptr_infos_test.pkl
├── simv2i_maptr_map_gt_train.json
├── simv2i_maptr_map_gt_val.json
├── simv2i_maptr_map_gt_test.json
└── image folders referenced by the PKL files
```

The public SimV2I-HD configs read `SIMV2I_HD_ROOT` directly. For backward
compatibility with the original experiment workspace, they also fall back to
`data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k` when the environment
variable is not set. A symlink under `data/maptr/` can be used for that legacy
layout, but storing the dataset inside this Git repository is not required.

During evaluation, MapTR writes an additional generated GT cache under
`outputs/maptr/eval_cache/` through the config field `map_ann_file`. This cache
is built automatically from the split PKL annotations when it is missing; it is
not a dataset file and is intentionally ignored by git.

See [DATA.md](DATA.md) for the expected layout and validation commands.

## Main Paper Configs

| Setting | Config |
| --- | --- |
| Ego-only | `integrations/maptr/configs/simv2i_maptr_ego_only_r18_20k_b4.py` |
| Dynamic Top-4 RSU | `integrations/maptr/configs/simv2i_maptr_v2_dynamic_rsu_top4_r18_20k.py` |
| Pose-gated Top-4 V2I | `integrations/maptr/configs/simv2i_maptr_pose_gated_v2i_r18_20k_b4.py` |
| Pose-gated Top-2 V2I | `integrations/maptr/configs/simv2i_maptr_pose_gated_v2i_r18_20k_top2_b4.py` |

The public wrapper scripts call the existing experiment scripts without
changing model, dataset, training, or evaluation behavior:

```bash
bash scripts/train_ego_only.sh
bash scripts/train_dynamic_top4.sh
bash scripts/train_pose_gated_top4.sh

bash scripts/eval_ego_only.sh path/to/checkpoint.pth
bash scripts/eval_dynamic_top4.sh path/to/checkpoint.pth
bash scripts/eval_pose_gated_top4.sh path/to/checkpoint.pth
```

For full reproduction notes, see [REPRODUCE.md](REPRODUCE.md).

## Useful Lightweight Checks

Validate the dataset structure:

```bash
bash scripts/validate_server_maptr_data.sh \
  --dataset-dir "${SIMV2I_HD_ROOT}" \
  --require-data
```

Check that Python files compile:

```bash
python -m py_compile tools/export_pose_gate_stats.py tools/plot_qualitative_cases.py
```

## Analysis And Visualization

- `tools/visualize_simv2i_maptr_predictions.py`: BEV qualitative figures.
- `tools/plot_qualitative_cases.py`: supplementary GT/Ego/Dynamic/Pose-gated
  comparison cases.
- `tools/export_pose_gate_stats.py`: inference-only pose-gate statistics.
- `tools/collect_method_overview_assets.py`: simulator-derived visual asset
  collection for method figures.

These utilities are analysis-only unless explicitly used as part of a run.

## Acknowledgements

This codebase builds on the original MapTR project and its legacy
MMDetection3D/OpenMMLab dependencies. Please also follow the license and
citation requirements of the upstream MapTR and MMDetection3D projects.
