# Reproducing The Paper Experiments

This document records the public entry points for the SimV2I-HD MapTR
experiments. It does not change the original paper configs and does not require
committing datasets, checkpoints, prediction PKLs, or generated outputs.

## 1. Prepare Runtime

```bash
export MAPTR_ROOT=/path/to/MapTR-SimV2I-HD
export SIMV2I_HD_ROOT=/path/to/simv2i_hd_benchmark_v2_dynamic_rsu_20k
cd "${MAPTR_ROOT}"

conda activate maptr_simv2i
export MAPTR_PYTHON=python
export PYTHONPATH="${MAPTR_ROOT}:${MAPTR_ROOT}/mmdetection3d:${PYTHONPATH:-}"
```

See [INSTALL.md](INSTALL.md) for environment creation and extension builds.

## 2. Prepare Data

Prepare the scenario-disjoint controlled 20k dynamic-RSU MapTR-format split
outside the git repository and export its root:

```bash
export SIMV2I_HD_ROOT=/path/to/simv2i_hd_benchmark_v2_dynamic_rsu_20k
```

Then validate:

```bash
bash scripts/validate_server_maptr_data.sh \
  --dataset-dir "${SIMV2I_HD_ROOT}" \
  --require-data
```

The public configs use `SIMV2I_HD_ROOT` directly and do not require editing
config files. For backward compatibility only, the same configs also fall back
to `data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k` if the environment
variable is unset. See [DATA.md](DATA.md) for the full expected layout and the
optional symlink workflow.

For the geo subset configs, keep the annotation root and image root separate:

```bash
export SIMV2I_HD_GEO_ROOT=/path/to/simv2i_hd_benchmark_v2_dynamic_rsu_20k_geo_subset
# Optional. Leave unset when PKL paths such as data/raw/... resolve from MAPTR_ROOT.
export SIMV2I_HD_IMAGE_ROOT=/path/whose/child/is/data/raw
```

The geo wrappers check that `SIMV2I_HD_GEO_ROOT` contains the split PKL/GT JSON
files and that `SIMV2I_HD_IMAGE_ROOT` or `.` can resolve sample image paths.
Symlinking `data/raw` under `${MAPTR_ROOT}` remains an optional compatibility
workaround.

## 3. Main Configs

| Experiment | Config | Default output directory |
| --- | --- | --- |
| Ego-only | `integrations/maptr/configs/simv2i_maptr_ego_only_r18_20k_b4.py` | `outputs/maptr/ego_only_r18_20k_b4` |
| Dynamic Top-4 RSU | `integrations/maptr/configs/simv2i_maptr_v2_dynamic_rsu_top4_r18_20k.py` | `outputs/maptr/v2_dynamic_rsu_top4_r18_20k` |
| Pose-gated Top-4 V2I | `integrations/maptr/configs/simv2i_maptr_pose_gated_v2i_r18_20k_b4.py` | `outputs/maptr/pose_gated_v2i_r18_20k_b4` |

## 4. Training Entry Points

These wrappers call the existing experiment scripts. They are provided for a
stable public interface; the underlying training logic is unchanged.

```bash
bash scripts/train_ego_only.sh
bash scripts/train_dynamic_top4.sh
bash scripts/train_pose_gated_top4.sh
```

Common optional variables:

```bash
export MAPTR_PYTHON=python
export MAPTR_GPUS=4
export GPUS=4
export PORT=29500
```

Use `MAPTR_TRAIN_WORK_DIR=/path/to/output` if you need a non-default output
directory.

## 5. Evaluation Entry Points

Run evaluation from existing checkpoints:

```bash
bash scripts/eval_ego_only.sh path/to/ego_checkpoint.pth
bash scripts/eval_dynamic_top4.sh path/to/dynamic_top4_checkpoint.pth
bash scripts/eval_pose_gated_top4.sh path/to/pose_gated_checkpoint.pth
```

Use `MAPTR_EVAL_WORK_DIR=/path/to/output` to redirect evaluation logs/results.

The `map_ann_file=outputs/maptr/eval_cache/...` entries in the configs are
generated evaluation caches, not required dataset files. On the first
evaluation run, MapTR creates the cache from the split PKL annotations loaded
from `SIMV2I_HD_ROOT`. If the dataset root changes, delete the corresponding
file under `outputs/maptr/eval_cache/` and rerun evaluation to regenerate it.

## 6. Analysis Utilities

Pose-gate analysis, inference-only:

```bash
"${MAPTR_PYTHON:-python}" tools/export_pose_gate_stats.py \
  --config integrations/maptr/configs/simv2i_maptr_pose_gated_v2i_r18_20k_b4.py \
  --checkpoint path/to/pose_gated_checkpoint.pth \
  --split val \
  --output-dir outputs/gate_analysis
```

Supplementary qualitative cases from saved predictions:

```bash
"${MAPTR_PYTHON:-python}" tools/plot_qualitative_cases.py \
  --data-root . \
  --split test \
  --ego-pred path/to/ego_predictions.pkl \
  --dynamic-top4-pred path/to/dynamic_predictions.pkl \
  --pose-gated-pred path/to/pose_gated_predictions.pkl \
  --output-dir outputs/qualitative_cases \
  --num-cases 5
```

Paper-ready visualization:

```bash
bash scripts/make_paper_ready_sample3608_figure.sh
```

## 7. Reproducibility Notes

- No dataset, checkpoint, prediction PKL, or generated output is tracked in git.
- Public wrappers keep existing config names and experiment scripts intact.
- Public configs use `SIMV2I_HD_ROOT` with a backward-compatible
  `data/maptr/...` fallback.
- Use small smoke configs or `--max-samples` options for quick checks; do not
  start full training unless intentionally reproducing the paper runs.
