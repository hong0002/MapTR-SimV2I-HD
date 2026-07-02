#!/usr/bin/env python3
"""Export pose-aware RSU gate values from a trained Pose-gated V2I checkpoint.

This is an inference-only analysis utility. It loads the trained
PoseAwareGatedRSUFusion weights from a checkpoint, evaluates the gate MLP on
real validation/test metadata, and writes CSV summaries, figures, and a short
Markdown report. It does not train the model and does not run or save MapTR
predictions.

Example:
    export MAPTR_ROOT=/path/to/MapTR
    export MAPTR_PYTHON=/path/to/conda/envs/maptr_simv2i/bin/python
    env PYTHONPATH="${MAPTR_ROOT}:${MAPTR_ROOT}/mmdetection3d" \
      "${MAPTR_PYTHON}" \
      tools/export_pose_gate_stats.py \
      --config integrations/maptr/configs/simv2i_maptr_pose_gated_v2i_r18_20k_b4.py \
      --checkpoint outputs/maptr/pose_gated_v2i_r18_20k_b4/epoch_24.pth \
      --split val \
      --output-dir outputs/gate_analysis
"""

import argparse
import copy
import csv
import importlib
import math
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, REPO_ROOT / "mmdetection3d"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/maptr_mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/maptr_xdg_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mmcv import Config
from mmcv.utils import import_modules_from_strings
from mmdet3d.datasets import build_dataset

from projects.mmdet3d_plugin.maptr.modules.pose_gated_rsu_fusion import (  # noqa: E402
    PoseAwareGatedRSUFusion,
)


QUALITY_TO_SCORE = {
    "pass": 1.0,
    "warning": 0.5,
    "low_quality": 0.0,
    "bad": 0.0,
}

REQUIRED_COLUMNS = [
    "sample_id",
    "split",
    "rsu_slot",
    "physical_rsu_id",
    "gate_value",
    "distance_to_ego",
    "relative_yaw",
    "coverage_score",
    "visual_quality_score",
    "ego_x",
    "ego_y",
    "rsu_x",
    "rsu_y",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export pose-aware RSU gate statistics without training."
    )
    parser.add_argument("--config", required=True, help="Pose-gated V2I config.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Trained Pose-gated V2I checkpoint.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=("val", "test"),
        help="Dataset split to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/gate_analysis",
        help="Directory for CSVs, figures, and report.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for the lightweight gate module.",
    )
    parser.add_argument(
        "--raw-root",
        default="data/raw",
        help="Root used to find optional dynamic RSU QA metadata.",
    )
    return parser.parse_args()


def import_config_plugins(cfg, config_path):
    if cfg.get("custom_imports", None):
        import_modules_from_strings(**cfg["custom_imports"])

    if not getattr(cfg, "plugin", False):
        return

    if hasattr(cfg, "plugin_dir"):
        module_dir = os.path.dirname(cfg.plugin_dir)
    else:
        module_dir = os.path.dirname(config_path)
    module_parts = [part for part in module_dir.split("/") if part]
    if module_parts:
        importlib.import_module(".".join(module_parts))


def build_split_dataset(cfg, split):
    dataset_cfg = copy.deepcopy(cfg.data[split])
    if isinstance(dataset_cfg, (list, tuple)):
        raise TypeError("This exporter expects a single dataset config.")
    dataset_cfg.test_mode = True
    dataset_cfg.pop("samples_per_gpu", None)
    dataset_cfg.pop("workers_per_gpu", None)
    dataset_cfg.map_ann_file = None
    return build_dataset(dataset_cfg)


def load_pose_gate_from_checkpoint(cfg, checkpoint_path, device):
    model_cfg = cfg.get("model", {})
    pose_gate_cfg = copy.deepcopy(model_cfg.get("pose_gate", {}))
    pose_gate_cfg.pop("type", None)
    pose_gate_cfg.setdefault("ego_view_count", model_cfg.get("ego_view_count", 6))
    pose_gate_cfg.setdefault("rsu_view_count", model_cfg.get("rsu_view_count", 4))
    gate = PoseAwareGatedRSUFusion(**pose_gate_cfg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a state_dict-like mapping.")

    gate_state = {}
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if clean_key.startswith("pose_gate."):
            gate_state[clean_key[len("pose_gate.") :]] = value

    if not gate_state and gate.gate_mode == "pose":
        raise RuntimeError(
            "No pose_gate.* parameters were found in checkpoint: {}".format(
                checkpoint_path
            )
        )

    missing, unexpected = gate.load_state_dict(gate_state, strict=False)
    gate.to(device)
    gate.eval()
    return gate, sorted(missing), sorted(unexpected)


def split_list_field(value):
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [item.strip().strip("'\"") for item in text.split(",") if item.strip()]


def split_float_field(value):
    values = []
    for item in split_list_field(value):
        try:
            values.append(float(item))
        except ValueError:
            values.append(float("nan"))
    return values


def quality_score(value):
    if value is None:
        return float("nan")
    text = str(value).strip().lower()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return QUALITY_TO_SCORE.get(text, float("nan"))


def load_dynamic_rsu_qa(raw_root):
    raw_root = Path(raw_root)
    lookup = {}
    if not raw_root.exists():
        return lookup

    for csv_path in raw_root.glob("*/dynamic_rsu_qa/dynamic_rsu_qa_by_frame.csv"):
        with csv_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                run_id = row.get("run_id") or csv_path.parents[1].name
                try:
                    frame = int(float(row.get("frame", "")))
                except ValueError:
                    continue
                rsu_ids = split_list_field(row.get("selected_rsu_ids"))
                geom_scores = split_float_field(row.get("selected_geometry_scores"))
                final_scores = split_float_field(row.get("selected_final_scores"))
                qualities = split_list_field(row.get("selected_visual_quality"))
                for slot, rsu_id in enumerate(rsu_ids):
                    coverage = value_at(geom_scores, slot)
                    if not math.isfinite(coverage):
                        coverage = value_at(final_scores, slot)
                    lookup[(run_id, frame, rsu_id)] = {
                        "coverage_score": coverage,
                        "visual_quality_score": quality_score(
                            value_at(qualities, slot, default="")
                        ),
                    }

    for csv_path in raw_root.glob("*/coverage_qa/coverage_by_frame.csv"):
        run_id = csv_path.parents[1].name
        with csv_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    frame = int(float(row.get("frame", "")))
                except ValueError:
                    continue
                for key, value in row.items():
                    if not key.startswith("valid_ratio_"):
                        continue
                    rsu_id = key[len("valid_ratio_") :]
                    entry = lookup.setdefault(
                        (run_id, frame, rsu_id),
                        {
                            "coverage_score": float("nan"),
                            "visual_quality_score": float("nan"),
                        },
                    )
                    if not math.isfinite(entry.get("coverage_score", float("nan"))):
                        entry["coverage_score"] = finite_or_nan(value)
    return lookup


def value_at(values, index, default=float("nan")):
    if values is None or index >= len(values):
        return default
    return values[index]


def finite_or_nan(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def nested_location_xy(obj):
    if not isinstance(obj, dict):
        return float("nan"), float("nan")
    for key in (
        "fixed_world_transform",
        "camera_world_transform",
        "sensor_mount_transform",
        "sensor_mount_transform_config",
    ):
        transform = obj.get(key)
        if not isinstance(transform, dict):
            continue
        loc = transform.get("location")
        if isinstance(loc, dict):
            return finite_or_nan(loc.get("x")), finite_or_nan(loc.get("y"))
    return float("nan"), float("nan")


def parse_run_frame_from_path(path):
    if not path:
        return None, None
    normalized = str(path).replace("\\", "/")
    match = re.search(r"(data/raw/)?(?P<run>simv2i_[^/]+)/sensors/.+/(?P<frame>\d+)\.[^.]+$", normalized)
    if not match:
        return None, None
    return match.group("run"), int(match.group("frame"))


def wrap_degrees(angle):
    angle = finite_or_nan(angle)
    if not math.isfinite(angle):
        return float("nan")
    return (angle + 180.0) % 360.0 - 180.0


def matrix_distance_yaw(matrix):
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (4, 4):
        return float("nan"), float("nan")
    translation = arr[:3, 3]
    distance = float(np.linalg.norm(translation[:2]))
    yaw = math.degrees(math.atan2(float(arr[1, 0]), float(arr[0, 0])))
    return distance, wrap_degrees(yaw)


def ego_xy_from_info(raw_info, data_info):
    ego_translation = raw_info.get("ego2global_translation")
    if ego_translation is None:
        ego_translation = data_info.get("ego2global_translation")
    if ego_translation is not None and len(ego_translation) >= 2:
        return finite_or_nan(ego_translation[0]), finite_or_nan(ego_translation[1])

    ego_pose = raw_info.get("ego_pose", {})
    location = ego_pose.get("location", {}) if isinstance(ego_pose, dict) else {}
    return finite_or_nan(location.get("x")), finite_or_nan(location.get("y"))


def quaternion_to_matrix(value):
    quat = np.asarray(value, dtype=np.float64)
    if quat.shape != (4,):
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quat
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def derive_global_xy_from_camera2ego(data_info, view_idx):
    matrices = data_info.get("camera2ego") or []
    if view_idx >= len(matrices):
        return float("nan"), float("nan")
    camera2ego = np.asarray(matrices[view_idx], dtype=np.float64)
    if camera2ego.shape != (4, 4):
        return float("nan"), float("nan")
    ego2global = np.eye(4, dtype=np.float64)
    ego_translation = data_info.get("ego2global_translation")
    ego_rotation = data_info.get("ego2global_rotation")
    if ego_translation is not None and len(ego_translation) >= 3:
        ego2global[:3, 3] = np.asarray(ego_translation[:3], dtype=np.float64)
    if ego_rotation is not None:
        rotation = np.asarray(ego_rotation, dtype=np.float64)
        if rotation.shape == (3, 3):
            ego2global[:3, :3] = rotation
        elif rotation.shape == (4,):
            ego2global[:3, :3] = quaternion_to_matrix(rotation)
    camera_global = ego2global @ camera2ego
    return finite_or_nan(camera_global[0, 3]), finite_or_nan(camera_global[1, 3])


def build_rows_for_sample(gate, data_info, split, qa_lookup, device):
    selected_names = list(data_info.get("selected_camera_names", []))
    selected_groups = list(data_info.get("selected_camera_groups", []))
    view_count = len(selected_names)
    if view_count <= gate.ego_view_count:
        return []

    dummy = torch.ones((1, view_count, 1, 1, 1), device=device)
    with torch.no_grad():
        _, stats = gate(
            [dummy],
            [data_info],
            collect_stats=True,
            return_details=True,
        )

    gate_values = stats.get("gate_values")
    if not gate_values:
        raise RuntimeError("Gate details were not returned by PoseAwareGatedRSUFusion.")
    gates = np.asarray(gate_values[0], dtype=np.float64)

    raw_info = data_info.get("_raw_info", {})
    sample_id = str(
        data_info.get("sample_idx")
        or raw_info.get("sample_id")
        or raw_info.get("token")
        or ""
    )
    ego_x, ego_y = ego_xy_from_info(raw_info, data_info)
    camera2ego = data_info.get("camera2ego") or []
    raw_rsu_cams = raw_info.get("rsu_cams", {})

    rows = []
    for rsu_slot, gate_value in enumerate(gates, start=1):
        view_idx = gate.ego_view_count + rsu_slot - 1
        physical_rsu_id = (
            selected_names[view_idx]
            if view_idx < len(selected_names)
            else "rsu_slot_{}".format(rsu_slot)
        )
        camera_group = (
            selected_groups[view_idx] if view_idx < len(selected_groups) else "rsu"
        )
        if str(camera_group).lower() != "rsu":
            continue

        cam_info = raw_rsu_cams.get(physical_rsu_id, {})
        data_path = cam_info.get("data_path") or cam_info.get("img_path") or ""
        run_id, frame = parse_run_frame_from_path(data_path)
        run_id = run_id or raw_info.get("run_id") or raw_info.get("scene_token")

        distance, yaw = matrix_distance_yaw(value_at(camera2ego, view_idx, None))
        distance = finite_or_nan(cam_info.get("distance_to_ego_m", distance))
        yaw = wrap_degrees(cam_info.get("yaw_difference_to_ego_deg", yaw))

        rsu_x, rsu_y = nested_location_xy(cam_info)
        if not math.isfinite(rsu_x) or not math.isfinite(rsu_y):
            rsu_x, rsu_y = derive_global_xy_from_camera2ego(data_info, view_idx)

        qa = qa_lookup.get((run_id, frame, physical_rsu_id), {})
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "rsu_slot": rsu_slot,
                "physical_rsu_id": physical_rsu_id,
                "gate_value": finite_or_nan(gate_value),
                "distance_to_ego": distance,
                "relative_yaw": yaw,
                "coverage_score": finite_or_nan(qa.get("coverage_score")),
                "visual_quality_score": finite_or_nan(
                    qa.get("visual_quality_score")
                ),
                "ego_x": ego_x,
                "ego_y": ego_y,
                "rsu_x": rsu_x,
                "rsu_y": rsu_y,
            }
        )
    return rows


def gate_distribution_stats(df):
    gate = df["gate_value"].dropna()
    return {
        "count": int(gate.shape[0]),
        "mean": gate.mean(),
        "std": gate.std(),
        "min": gate.min(),
        "max": gate.max(),
        "median": gate.median(),
    }


def compute_correlations(df):
    rows = []
    try:
        from scipy import stats as scipy_stats
    except Exception:
        scipy_stats = None

    for column in (
        "distance_to_ego",
        "relative_yaw",
        "coverage_score",
        "visual_quality_score",
    ):
        valid = (
            df[["gate_value", column]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        row = {
            "metadata": column,
            "n": int(valid.shape[0]),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
            "method": "insufficient_data",
        }
        if (
            valid.shape[0] >= 3
            and valid["gate_value"].nunique(dropna=True) > 1
            and valid[column].nunique(dropna=True) > 1
        ):
            if scipy_stats is not None:
                pearson = scipy_stats.pearsonr(valid[column], valid["gate_value"])
                spearman = scipy_stats.spearmanr(valid[column], valid["gate_value"])
                row.update(
                    pearson_r=float(pearson.statistic),
                    pearson_p=float(pearson.pvalue),
                    spearman_r=float(spearman.statistic),
                    spearman_p=float(spearman.pvalue),
                    method="scipy",
                )
            else:
                row.update(
                    pearson_r=float(valid[column].corr(valid["gate_value"], method="pearson")),
                    spearman_r=float(valid[column].corr(valid["gate_value"], method="spearman")),
                    method="pandas",
                )
        rows.append(row)
    return pd.DataFrame(rows)


def ensure_required_columns(df):
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    return df[REQUIRED_COLUMNS]


def save_distribution_plot(df, out_path):
    values = df["gate_value"].dropna()
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=160)
    ax.hist(values, bins=32, color="#386cb0", alpha=0.82, edgecolor="white")
    ax.set_xlabel("Gate value")
    ax.set_ylabel("Count")
    ax.set_title("Pose-aware RSU Gate Distribution")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_slot_plot(df, out_path):
    slots = sorted(df["rsu_slot"].dropna().unique())
    data = [df.loc[df["rsu_slot"] == slot, "gate_value"].dropna() for slot in slots]
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=160)
    ax.boxplot(data, labels=[str(int(slot)) for slot in slots], showfliers=False)
    ax.set_xlabel("Selected RSU slot")
    ax.set_ylabel("Gate value")
    ax.set_title("Gate by Dynamic Top-k Slot")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_scatter_plot(df, column, xlabel, out_path):
    valid = (
        df[[column, "gate_value"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=160)
    if valid.empty:
        ax.text(
            0.5,
            0.5,
            "No metadata available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        ax.scatter(valid[column], valid["gate_value"], s=10, alpha=0.42, color="#1b9e77")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Gate value")
    ax.set_title("Gate vs {}".format(xlabel))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_physical_rsu_plot(physical_summary, out_path):
    top = physical_summary.sort_values("count", ascending=False).head(20)
    top = top.sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=160)
    if top.empty:
        ax.text(
            0.5,
            0.5,
            "No physical RSU data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        ax.bar(
            top.index.astype(str),
            top["mean"],
            yerr=top["std"].fillna(0.0),
            color="#984ea3",
            alpha=0.82,
            capsize=2,
        )
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    ax.set_xlabel("Physical RSU id")
    ax.set_ylabel("Mean gate value")
    ax.set_title("Gate by Frequently Selected Physical RSU")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fmt(value, digits=4):
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        return "{:.{}f}".format(float(value), digits)
    except (TypeError, ValueError):
        return str(value)


def write_report(
    out_path,
    args,
    dataset_len,
    processed_samples,
    df,
    slot_summary,
    physical_summary,
    corr_summary,
    dist_stats,
    missing,
    unexpected,
    figure_paths,
    csv_paths,
):
    per_sample_std = df.groupby("sample_id")["gate_value"].std().dropna()
    sample_std_mean = per_sample_std.mean() if not per_sample_std.empty else np.nan
    sample_std_max = per_sample_std.max() if not per_sample_std.empty else np.nan
    slot_mean_range = (
        slot_summary["mean"].max() - slot_summary["mean"].min()
        if not slot_summary.empty
        else np.nan
    )
    physical_mean_range = (
        physical_summary["mean"].max() - physical_summary["mean"].min()
        if not physical_summary.empty
        else np.nan
    )

    if dist_stats["std"] > 1e-4 or sample_std_mean > 1e-4:
        variation_note = (
            "Gate values vary across samples/RSU selections, which is evidence "
            "against a purely constant global down-weighting for this checkpoint."
        )
    else:
        variation_note = (
            "Gate values are nearly constant in this export; this run alone does "
            "not show strong sample-specific modulation."
        )

    lines = [
        "# Pose-aware RSU Gate Analysis",
        "",
        "This report was generated by `tools/export_pose_gate_stats.py`.",
        "It is inference-only: the model was set to `eval`, evaluated under `torch.no_grad()`, and no training, prediction export, dataset rewrite, or checkpoint update was performed.",
        "",
        "## Run",
        "",
        "- Config: `{}`".format(args.config),
        "- Checkpoint: `{}`".format(args.checkpoint),
        "- Split: `{}`".format(args.split),
        "- Dataset samples available: `{}`".format(dataset_len),
        "- Samples processed: `{}`".format(processed_samples),
        "- Gate rows exported: `{}`".format(len(df)),
        "- Device: `{}`".format(args.device),
        "- Missing gate checkpoint keys: `{}`".format(", ".join(missing) if missing else "none"),
        "- Unexpected gate checkpoint keys: `{}`".format(", ".join(unexpected) if unexpected else "none"),
        "",
        "## Gate Distribution",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in ("count", "mean", "std", "min", "max", "median"):
        lines.append("| {} | {} |".format(key, fmt(dist_stats[key])))

    lines.extend(
        [
            "",
            "## Variation Summary",
            "",
            "- Mean within-sample gate std: `{}`".format(fmt(sample_std_mean)),
            "- Max within-sample gate std: `{}`".format(fmt(sample_std_max)),
            "- Range of slot mean gates: `{}`".format(fmt(slot_mean_range)),
            "- Range of physical-RSU mean gates: `{}`".format(fmt(physical_mean_range)),
            "- Interpretation: {}".format(variation_note),
            "",
            "## Gate by RSU Slot",
            "",
            "| rsu_slot | count | mean | std | min | max | median |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in slot_summary.reset_index().iterrows():
        lines.append(
            "| {rsu_slot} | {count} | {mean} | {std} | {min} | {max} | {median} |".format(
                rsu_slot=int(row["rsu_slot"]),
                count=int(row["count"]),
                mean=fmt(row["mean"]),
                std=fmt(row["std"]),
                min=fmt(row["min"]),
                max=fmt(row["max"]),
                median=fmt(row["median"]),
            )
        )

    lines.extend(
        [
            "",
            "## Correlations",
            "",
            "| metadata | n | Pearson r | Pearson p | Spearman rho | Spearman p | method |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in corr_summary.iterrows():
        lines.append(
            "| {metadata} | {n} | {pearson_r} | {pearson_p} | {spearman_r} | {spearman_p} | {method} |".format(
                metadata=row["metadata"],
                n=int(row["n"]),
                pearson_r=fmt(row["pearson_r"]),
                pearson_p=fmt(row["pearson_p"]),
                spearman_r=fmt(row["spearman_r"]),
                spearman_p=fmt(row["spearman_p"]),
                method=row["method"],
            )
        )

    lines.extend(
        [
            "",
            "## Physical RSU Summary",
            "",
            "Top frequently selected physical RSUs by count:",
            "",
            "| physical_rsu_id | count | mean | std |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for physical_id, row in physical_summary.sort_values("count", ascending=False).head(12).iterrows():
        lines.append(
            "| {} | {} | {} | {} |".format(
                physical_id,
                int(row["count"]),
                fmt(row["mean"]),
                fmt(row["std"]),
            )
        )

    metadata_coverage = {
        column: int(df[column].notna().sum())
        for column in (
            "distance_to_ego",
            "relative_yaw",
            "coverage_score",
            "visual_quality_score",
        )
    }
    lines.extend(
        [
            "",
            "## Metadata Coverage",
            "",
        ]
    )
    for column, count in metadata_coverage.items():
        lines.append("- `{}` non-null rows: `{}/{}`".format(column, count, len(df)))
    if (
        metadata_coverage.get("coverage_score", 0) == 0
        or metadata_coverage.get("visual_quality_score", 0) == 0
    ):
        lines.append(
            "- Note: coverage/visual-quality correlations are reported as "
            "`insufficient_data` when the selected split has no matching raw QA "
            "CSV metadata in this workspace."
        )

    lines.extend(["", "## Outputs", "", "CSV files:"])
    for path in csv_paths:
        lines.append("- `{}`".format(path))
    lines.extend(["", "Figures:"])
    for path in figure_paths:
        lines.append("- `{}`".format(path))

    out_path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    import_config_plugins(cfg, args.config)
    dataset = build_split_dataset(cfg, args.split)
    gate, missing, unexpected = load_pose_gate_from_checkpoint(
        cfg,
        args.checkpoint,
        args.device,
    )
    qa_lookup = load_dynamic_rsu_qa(args.raw_root)

    dataset_len = len(dataset)
    sample_count = dataset_len
    if args.max_samples is not None:
        sample_count = min(dataset_len, max(0, args.max_samples))

    rows = []
    for index in range(sample_count):
        data_info = dataset.get_data_info(index)
        rows.extend(
            build_rows_for_sample(
                gate,
                data_info,
                args.split,
                qa_lookup,
                args.device,
            )
        )
        if (index + 1) % 250 == 0 or index + 1 == sample_count:
            print("Processed {}/{} samples".format(index + 1, sample_count))

    if not rows:
        raise RuntimeError("No RSU gate rows were exported.")

    df = ensure_required_columns(pd.DataFrame(rows))
    csv_paths = []
    gate_values_path = output_dir / "gate_values_per_rsu.csv"
    df.to_csv(gate_values_path, index=False)
    csv_paths.append(gate_values_path)

    slot_summary = (
        df.groupby("rsu_slot")["gate_value"]
        .agg(["count", "mean", "std", "min", "max", "median"])
        .reset_index()
    )
    slot_summary_path = output_dir / "gate_slot_summary.csv"
    slot_summary.to_csv(slot_summary_path, index=False)
    csv_paths.append(slot_summary_path)

    physical_summary = (
        df.groupby("physical_rsu_id")["gate_value"]
        .agg(["count", "mean", "std", "min", "max", "median"])
        .sort_values("count", ascending=False)
    )
    physical_summary_path = output_dir / "gate_physical_rsu_summary.csv"
    physical_summary.to_csv(physical_summary_path)
    csv_paths.append(physical_summary_path)

    corr_summary = compute_correlations(df)
    corr_summary_path = output_dir / "gate_correlation_summary.csv"
    corr_summary.to_csv(corr_summary_path, index=False)
    csv_paths.append(corr_summary_path)

    figure_paths = [
        figures_dir / "gate_distribution.png",
        figures_dir / "gate_by_slot.png",
        figures_dir / "gate_vs_distance.png",
        figures_dir / "gate_vs_yaw.png",
        figures_dir / "gate_vs_coverage.png",
        figures_dir / "gate_by_physical_rsu.png",
    ]
    save_distribution_plot(df, figure_paths[0])
    save_slot_plot(df, figure_paths[1])
    save_scatter_plot(df, "distance_to_ego", "Distance to ego (m)", figure_paths[2])
    save_scatter_plot(df, "relative_yaw", "Relative yaw (deg)", figure_paths[3])
    save_scatter_plot(df, "coverage_score", "Coverage score", figure_paths[4])
    save_physical_rsu_plot(physical_summary, figure_paths[5])

    report_path = output_dir / "gate_analysis.md"
    write_report(
        report_path,
        args,
        dataset_len,
        sample_count,
        df,
        slot_summary,
        physical_summary,
        corr_summary,
        gate_distribution_stats(df),
        missing,
        unexpected,
        figure_paths,
        csv_paths,
    )

    print("Wrote {}".format(gate_values_path))
    print("Wrote {}".format(slot_summary_path))
    print("Wrote {}".format(corr_summary_path))
    print("Wrote {}".format(report_path))
    print("Wrote figures to {}".format(figures_dir))


if __name__ == "__main__":
    main()
