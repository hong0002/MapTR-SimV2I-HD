#!/usr/bin/env python
"""Visualize MapTR prediction/GT coordinate alignment for SimV2I-HD."""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MAP_CLASSES = ("divider", "boundary", "ped_crossing")
CLASS_COLORS = {
    "divider": "#1f77b4",
    "boundary": "#ff7f0e",
    "ped_crossing": "#2ca02c",
}
TRANSFORMS = (
    "original",
    "swap_xy",
    "flip_y",
    "flip_x",
    "swap_xy_flip_y",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-json",
        default="outputs/maptr/ego_r18_stronger/eval/results/pts_bbox/nuscmap_results.json",
    )
    parser.add_argument(
        "--gt-json",
        default="outputs/maptr/eval_cache/simv2i_hd_v1_test_gt.json",
    )
    parser.add_argument(
        "--out-dir", default="outputs/maptr/debug_vis_coord_check"
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument(
        "--score-thresholds", default="all,0.3,0.5,0.7"
    )
    parser.add_argument(
        "--pc-range", nargs=4, type=float, default=(-15.0, -30.0, 15.0, 30.0)
    )
    parser.add_argument("--topk-labels", type=int, default=3)
    return parser.parse_args()


def sanitize_token(token):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", token)


def load_predictions(path):
    data = json.loads(Path(path).read_text())
    return data["results"]


def load_gts(path):
    data = json.loads(Path(path).read_text())
    return data["GTs"]


def vector_class(vector):
    if "cls_name" in vector:
        return vector["cls_name"]
    if "type" in vector:
        return MAP_CLASSES[int(vector["type"])]
    raise KeyError("Vector has neither cls_name nor type.")


def vector_score(vector):
    return float(vector.get("confidence_level", vector.get("score", 1.0)))


def vector_points(vector):
    pts = np.asarray(vector["pts"], dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("Bad points shape: {}".format(pts.shape))
    return pts[:, :2]


def transform_points(points, name):
    x = points[:, 0]
    y = points[:, 1]
    if name == "original":
        out = np.stack([x, y], axis=1)
    elif name == "swap_xy":
        out = np.stack([y, x], axis=1)
    elif name == "flip_y":
        out = np.stack([x, -y], axis=1)
    elif name == "flip_x":
        out = np.stack([-x, y], axis=1)
    elif name == "swap_xy_flip_y":
        out = np.stack([y, -x], axis=1)
    else:
        raise KeyError(name)
    return out.astype(np.float32)


def flatten_points(samples, transform_name="original", min_score=None):
    by_class = defaultdict(list)
    scores = defaultdict(list)
    for sample in samples:
        for vector in sample["vectors"]:
            score = vector_score(vector)
            if min_score is not None and score < min_score:
                continue
            cls_name = vector_class(vector)
            pts = transform_points(vector_points(vector), transform_name)
            by_class[cls_name].append(pts)
            scores[cls_name].append(score)
    return by_class, scores


def concat_points(by_class):
    arrays = []
    for class_arrays in by_class.values():
        arrays.extend(class_arrays)
    if not arrays:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(arrays, axis=0)


def minmax(points):
    if len(points) == 0:
        return {"x": [None, None], "y": [None, None]}
    return {
        "x": [float(points[:, 0].min()), float(points[:, 0].max())],
        "y": [float(points[:, 1].min()), float(points[:, 1].max())],
    }


def outside_ratio(points, pc_range):
    if len(points) == 0:
        return 0.0
    xmin, ymin, xmax, ymax = pc_range
    outside = (
        (points[:, 0] < xmin)
        | (points[:, 0] > xmax)
        | (points[:, 1] < ymin)
        | (points[:, 1] > ymax)
    )
    return float(outside.mean())


def score_summary(values):
    if not values:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def nearest_mean_distance(pred_by_class, gt_by_class):
    distances = []
    for cls_name in MAP_CLASSES:
        pred_arrays = pred_by_class.get(cls_name, [])
        gt_arrays = gt_by_class.get(cls_name, [])
        if not pred_arrays or not gt_arrays:
            continue
        pred = np.concatenate(pred_arrays, axis=0)
        gt = np.concatenate(gt_arrays, axis=0)
        if len(pred) == 0 or len(gt) == 0:
            continue
        # Chunk to avoid one large [N, M] allocation.
        chunk_mins = []
        for start in range(0, len(pred), 1024):
            p = pred[start : start + 1024]
            diff = p[:, None, :] - gt[None, :, :]
            dists = np.sqrt((diff * diff).sum(axis=2))
            chunk_mins.append(dists.min(axis=1))
        nearest = np.concatenate(chunk_mins, axis=0)
        distances.extend(nearest.tolist())
    if not distances:
        return {"mean": None, "median": None, "count": 0}
    arr = np.asarray(distances, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "count": int(arr.size),
    }


def plot_sample(sample_pred, sample_gt, transform_name, threshold, pc_range, out_file, topk):
    fig, ax = plt.subplots(figsize=(7, 10))
    xmin, ymin, xmax, ymax = pc_range
    ax.set_xlim(xmin - 2, xmax + 2)
    ax.set_ylim(ymin - 2, ymax + 2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        "{} | threshold {} | token {}".format(
            transform_name, threshold, sample_pred["sample_token"]
        ),
        fontsize=9,
    )
    ax.add_patch(
        plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor="black",
            linewidth=1.0,
        )
    )

    for vector in sample_gt["vectors"]:
        cls_name = vector_class(vector)
        pts = vector_points(vector)
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=CLASS_COLORS.get(cls_name, "gray"),
            linewidth=2.3,
            alpha=0.9,
            solid_capstyle="round",
        )

    kept_pred = []
    for vector in sample_pred["vectors"]:
        score = vector_score(vector)
        if threshold != "all" and score < float(threshold):
            continue
        cls_name = vector_class(vector)
        pts = transform_points(vector_points(vector), transform_name)
        kept_pred.append((score, cls_name, pts))
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=CLASS_COLORS.get(cls_name, "gray"),
            linewidth=1.1 + 1.8 * score,
            alpha=0.25 + 0.65 * min(max(score, 0.0), 1.0),
            linestyle="--",
        )

    for rank, (score, cls_name, pts) in enumerate(
        sorted(kept_pred, key=lambda item: item[0], reverse=True)[:topk],
        start=1,
    ):
        ax.text(
            float(pts[0, 0]),
            float(pts[0, 1]),
            "#{:d} {:.2f} {}".format(rank, score, cls_name),
            fontsize=6,
            color=CLASS_COLORS.get(cls_name, "black"),
            bbox=dict(facecolor="white", alpha=0.55, edgecolor="none"),
        )

    handles = []
    labels = []
    for cls_name in MAP_CLASSES:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=CLASS_COLORS[cls_name],
                linewidth=2.3,
            )
        )
        labels.append("{} GT".format(cls_name))
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=CLASS_COLORS[cls_name],
                linewidth=1.8,
                linestyle="--",
            )
        )
        labels.append("{} pred".format(cls_name))
    ax.legend(handles, labels, fontsize=6, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_sample_grid(sample_pred, sample_gt, pc_range, out_file):
    fig, axes = plt.subplots(1, len(TRANSFORMS), figsize=(20, 5), sharex=True, sharey=True)
    xmin, ymin, xmax, ymax = pc_range
    for ax, transform_name in zip(axes, TRANSFORMS):
        ax.set_xlim(xmin - 2, xmax + 2)
        ax.set_ylim(ymin - 2, ymax + 2)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.2, alpha=0.3)
        ax.set_title(transform_name, fontsize=8)
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor="black",
                linewidth=0.8,
            )
        )
        for vector in sample_gt["vectors"]:
            cls_name = vector_class(vector)
            pts = vector_points(vector)
            ax.plot(pts[:, 0], pts[:, 1], color=CLASS_COLORS[cls_name], linewidth=2.0)
        for vector in sample_pred["vectors"]:
            cls_name = vector_class(vector)
            score = vector_score(vector)
            pts = transform_points(vector_points(vector), transform_name)
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=CLASS_COLORS[cls_name],
                linestyle="--",
                linewidth=0.8 + 1.4 * score,
                alpha=0.2 + 0.65 * min(max(score, 0.0), 1.0),
            )
    fig.suptitle(sample_pred["sample_token"], fontsize=10)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def make_markdown(summary, out_path):
    lines = []
    lines.append("# MapTR Coordinate Alignment Debug")
    lines.append("")
    lines.append("- Prediction JSON: `{}`".format(summary["pred_json"]))
    lines.append("- GT JSON: `{}`".format(summary["gt_json"]))
    lines.append("- Samples visualized: `{}`".format(len(summary["selected_samples"])))
    lines.append("- PC range: `{}`".format(summary["pc_range"]))
    lines.append("")
    lines.append("## Selected Samples")
    lines.append("")
    for item in summary["selected_samples"]:
        lines.append("- index `{index}` token `{token}`".format(**item))
    lines.append("")
    lines.append("## Coordinate Ranges")
    lines.append("")
    lines.append("| source | transform | x min | x max | y min | y max | outside ratio |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in summary["coord_ranges"]:
        mm = row["minmax"]
        lines.append(
            "| {source} | {transform} | {xmin} | {xmax} | {ymin} | {ymax} | {outside:.4f} |".format(
                source=row["source"],
                transform=row["transform"],
                xmin=fmt(mm["x"][0]),
                xmax=fmt(mm["x"][1]),
                ymin=fmt(mm["y"][0]),
                ymax=fmt(mm["y"][1]),
                outside=row["outside_ratio"],
            )
        )
    lines.append("")
    lines.append("## Prediction Counts And Scores")
    lines.append("")
    lines.append("| class | count | min | p05 | p25 | p50 | p75 | p95 | max | mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cls_name in MAP_CLASSES:
        stats = summary["score_summary"][cls_name]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                cls_name,
                stats["count"],
                fmt(stats["min"]),
                fmt(stats["p05"]),
                fmt(stats["p25"]),
                fmt(stats["p50"]),
                fmt(stats["p75"]),
                fmt(stats["p95"]),
                fmt(stats["max"]),
                fmt(stats["mean"]),
            )
        )
    lines.append("")
    lines.append("## Selected-Sample Nearest Distance")
    lines.append("")
    lines.append("Lower is better. Distances are prediction points to nearest same-class GT point.")
    lines.append("")
    lines.append("| transform | point count | mean | median |")
    lines.append("|---|---:|---:|---:|")
    for name, stats in summary["selected_nearest_distance"].items():
        lines.append(
            "| {} | {} | {} | {} |".format(
                name, stats["count"], fmt(stats["mean"]), fmt(stats["median"])
            )
        )
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- Grid overlays: `grids/*.png`")
    lines.append("- Per-transform overlays: `overlays/*.png`")
    lines.append("- Machine-readable summary: `summary.json`")
    lines.append("- Score rows: `score_stats.csv`")
    out_path.write_text("\n".join(lines) + "\n")


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return "{:.4f}".format(float(value))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    overlay_dir = out_dir / "overlays"
    grid_dir = out_dir / "grids"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    grid_dir.mkdir(parents=True, exist_ok=True)

    pred_samples = load_predictions(args.pred_json)
    gt_samples = load_gts(args.gt_json)
    gt_by_token = {sample["sample_token"]: sample for sample in gt_samples}

    selected_indices = np.linspace(
        0, len(pred_samples) - 1, min(args.num_samples, len(pred_samples)), dtype=int
    ).tolist()
    thresholds = [
        item.strip()
        for item in args.score_thresholds.split(",")
        if item.strip()
    ]

    raw_pred_by_class, scores_by_class = flatten_points(pred_samples)
    raw_gt_by_class, _ = flatten_points(gt_samples)
    all_pred_points = concat_points(raw_pred_by_class)
    all_gt_points = concat_points(raw_gt_by_class)

    coord_ranges = [
        {
            "source": "gt",
            "transform": "original",
            "minmax": minmax(all_gt_points),
            "outside_ratio": outside_ratio(all_gt_points, args.pc_range),
        }
    ]
    for transform_name in TRANSFORMS:
        pred_by_class, _ = flatten_points(pred_samples, transform_name=transform_name)
        points = concat_points(pred_by_class)
        coord_ranges.append(
            {
                "source": "pred",
                "transform": transform_name,
                "minmax": minmax(points),
                "outside_ratio": outside_ratio(points, args.pc_range),
            }
        )

    selected_pred = [pred_samples[i] for i in selected_indices]
    selected_gt = [gt_by_token[pred_samples[i]["sample_token"]] for i in selected_indices]
    selected_nearest = {}
    for transform_name in TRANSFORMS:
        pred_by_class, _ = flatten_points(selected_pred, transform_name=transform_name)
        gt_by_class, _ = flatten_points(selected_gt)
        selected_nearest[transform_name] = nearest_mean_distance(
            pred_by_class, gt_by_class
        )

    selected_items = []
    for index in selected_indices:
        sample_pred = pred_samples[index]
        sample_gt = gt_by_token[sample_pred["sample_token"]]
        token = sanitize_token(sample_pred["sample_token"])
        selected_items.append({"index": int(index), "token": sample_pred["sample_token"]})
        grid_file = grid_dir / "sample_{:03d}__{}__grid.png".format(index, token)
        plot_sample_grid(sample_pred, sample_gt, args.pc_range, grid_file)
        for transform_name in TRANSFORMS:
            for threshold in thresholds:
                suffix = "all" if threshold == "all" else "score_ge_{}".format(
                    str(threshold).replace(".", "p")
                )
                out_file = overlay_dir / "sample_{:03d}__{}__{}__{}.png".format(
                    index, token, transform_name, suffix
                )
                plot_sample(
                    sample_pred,
                    sample_gt,
                    transform_name,
                    threshold,
                    args.pc_range,
                    out_file,
                    args.topk_labels,
                )

    score_summary_by_class = {
        cls_name: score_summary(scores_by_class.get(cls_name, []))
        for cls_name in MAP_CLASSES
    }

    summary = {
        "pred_json": args.pred_json,
        "gt_json": args.gt_json,
        "out_dir": str(out_dir),
        "pc_range": list(args.pc_range),
        "selected_samples": selected_items,
        "coord_ranges": coord_ranges,
        "score_summary": score_summary_by_class,
        "selected_nearest_distance": selected_nearest,
        "thresholds": thresholds,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    make_markdown(summary, out_dir / "summary.md")

    with (out_dir / "score_stats.csv").open("w", newline="") as f:
        fieldnames = [
            "class",
            "count",
            "min",
            "p05",
            "p25",
            "p50",
            "p75",
            "p95",
            "max",
            "mean",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cls_name, stats in score_summary_by_class.items():
            row = {"class": cls_name}
            row.update(stats)
            writer.writerow(row)

    print("Output directory:", out_dir)
    print("Selected samples:")
    for item in selected_items:
        print("  {index:03d} {token}".format(**item))
    print("\nCoordinate ranges:")
    for row in coord_ranges:
        print(
            "  {source:4s} {transform:14s} x={x} y={y} outside={outside:.4f}".format(
                source=row["source"],
                transform=row["transform"],
                x=row["minmax"]["x"],
                y=row["minmax"]["y"],
                outside=row["outside_ratio"],
            )
        )
    print("\nPrediction score summary:")
    for cls_name, stats in score_summary_by_class.items():
        print("  {}: {}".format(cls_name, stats))
    print("\nSelected-sample nearest distances:")
    for transform_name, stats in selected_nearest.items():
        print("  {}: {}".format(transform_name, stats))


if __name__ == "__main__":
    main()
