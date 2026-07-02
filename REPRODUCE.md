# Reproducing The Paper Experiments

This document records the public entry points for the SimV2I-HD MapTR
experiments. It does not change the original paper configs and does not require
committing datasets, checkpoints, prediction PKLs, or generated outputs.

## 1. Prepare Runtime

```bash
export MAPTR_ROOT=/path/to/MapTR
cd "${MAPTR_ROOT}"

conda activate maptr_simv2i
export MAPTR_PYTHON=python
export PYTHONPATH="${MAPTR_ROOT}:${MAPTR_ROOT}/mmdetection3d:${PYTHONPATH:-}"
```

See [INSTALL.md](INSTALL.md) for environment creation and extension builds.

## 2. Prepare Data

Place or symlink the 20k dynamic-RSU MapTR annotation split at:

```text
data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k
```

Then validate:

```bash
bash scripts/validate_server_maptr_data.sh \
  --dataset-dir data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k \
  --require-data
```

See [DATA.md](DATA.md) for the full expected layout.

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
- The default paper configs expect repo-relative dataset paths.
- Use small smoke configs or `--max-samples` options for quick checks; do not
  start full training unless intentionally reproducing the paper runs.
