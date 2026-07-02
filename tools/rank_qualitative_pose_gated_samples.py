#!/usr/bin/env python
"""Rank SimV2I-HD MapTR qualitative candidates for Pose-gated V2I."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from visualize_simv2i_maptr_predictions import (
    CLASS_NAMES,
    load_gt_records,
    load_info_tokens,
    load_prediction_records,
    normalize_vector,
)

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - exercised only when scipy is absent.
    cKDTree = None


DEFAULT_GT_JSON = (
    "data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k/"
    "simv2i_maptr_map_gt_test.json"
)
DEFAULT_INFOS = (
    "data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k/"
    "simv2i_maptr_infos_test.pkl"
)
DEFAULT_PREDS = {
    "ego": "outputs/maptr/visualization_predictions/ego_only_test_epoch18.pkl",
    "dynamic": (
        "outputs/maptr/visualization_predictions/dynamic_top4_test_epoch18.pkl"
    ),
    "constant": (
        "outputs/maptr/visualization_predictions/constant_gate_test_epoch18.pkl"
    ),
    "pose": "outputs/maptr/visualization_predictions/pose_gated_top4_test_epoch18.pkl",
}
CSV_COLUMNS = [
    "rank",
    "sample_index",
    "sample_token",
    "scene_or_sample_name_if_available",
    "score",
    "ego_f1_mean",
    "dynamic_f1_mean",
    "constant_f1_mean",
    "pose_f1_mean",
    "pose_minus_dynamic",
    "pose_minus_ego",
    "gt_divider_count",
    "gt_boundary_count",
    "gt_ped_crossing_count",
    "ego_divider_f1",
    "dynamic_divider_f1",
    "constant_divider_f1",
    "pose_divider_f1",
    "ego_boundary_f1",
    "dynamic_boundary_f1",
    "constant_boundary_f1",
    "pose_boundary_f1",
    "ego_ped_crossing_f1",
    "dynamic_ped_crossing_f1",
    "constant_ped_crossing_f1",
    "pose_ped_crossing_f1",
    "recommended_visualization_command",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank qualitative Pose-gated V2I MapTR samples."
    )
    parser.add_argument("--gt-json", default=DEFAULT_GT_JSON)
    parser.add_argument("--infos", default=DEFAULT_INFOS)
    parser.add_argument("--ego-pred", default=DEFAULT_PREDS["ego"])
    parser.add_argument("--dynamic-pred", default=DEFAULT_PREDS["dynamic"])
    parser.add_argument("--constant-pred", default=DEFAULT_PREDS["constant"])
    parser.add_argument("--pose-pred", default=DEFAULT_PREDS["pose"])
    parser.add_argument(
        "--out-dir",
        default="outputs/maptr/visualizations/qualitative_ranking",
    )
    parser.add_argument("--score-thr", type=float, default=0.4)
    parser.add_argument("--dist-thr", type=float, default=1.5)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-gt-points", type=int, default=20)
    parser.add_argument("--prefer-classes", default="divider,ped_crossing")
    parser.add_argument("--sample-step", type=float, default=1.0)
    parser.add_argument("--max-points-per-line", type=int, default=40)
    return parser.parse_args()


def parse_classes(value):
    classes = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in classes if name not in CLASS_NAMES]
    if unknown:
        raise ValueError(f"Unknown preferred classes: {unknown}")
    return classes or list(CLASS_NAMES)


def by_token(records):
    result = {}
    for index, item in enumerate(records):
        token = item.get("sample_token") or item.get("token") or str(index)
        result[str(token)] = item
    return result


def info_name(info, token):
    if not info:
        return token
    scene = info.get("scene_token")
    frame = info.get("frame_idx")
    if scene is not None and frame is not None:
        return f"{scene}:{frame}"
    for key in ("maptr_gt_path", "lidar_path"):
        if info.get(key):
            return Path(info[key]).stem
    return token


def group_vectors(record, score_thr=None):
    grouped = {name: [] for name in CLASS_NAMES}
    for raw_vector in record.get("vectors", []):
        vector = normalize_vector(raw_vector)
        if vector is None or vector["cls_name"] not in grouped:
            continue
        if score_thr is not None and vector["score"] < score_thr:
            continue
        grouped[vector["cls_name"]].append(vector["pts"])
    return grouped


def sample_polyline(points, sample_step, max_points):
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)
    pts = pts[:, :2]
    if pts.shape[0] == 1:
        return np.repeat(pts, 2, axis=0)

    deltas = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(deltas, axis=1)
    keep = np.concatenate([[True], seg_lens > 1e-6])
    pts = pts[keep]
    if pts.shape[0] == 1:
        return np.repeat(pts, 2, axis=0)

    deltas = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cumulative[-1])
    if total <= 1e-6:
        return np.repeat(pts[:1], 2, axis=0)

    count = max(2, int(math.ceil(total / sample_step)) + 1)
    count = min(max_points, count)
    distances = np.linspace(0.0, total, count)
    xs = np.interp(distances, cumulative, pts[:, 0])
    ys = np.interp(distances, cumulative, pts[:, 1])
    return np.stack([xs, ys], axis=1)


def vectors_to_points(vectors, sample_step, max_points):
    sampled = [
        sample_polyline(points, sample_step, max_points)
        for points in vectors
    ]
    sampled = [points for points in sampled if len(points) >= 2]
    if not sampled:
        return np.zeros((0, 2), dtype=float)
    return np.concatenate(sampled, axis=0)


def nearest_distances(query_points, ref_points):
    if len(query_points) == 0 or len(ref_points) == 0:
        return np.array([], dtype=float)
    if cKDTree is not None:
        distances, _ = cKDTree(ref_points).query(query_points, k=1)
        return distances

    chunks = []
    for start in range(0, len(query_points), 512):
        query = query_points[start : start + 512]
        diff = query[:, None, :] - ref_points[None, :, :]
        chunks.append(np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1))
    return np.concatenate(chunks, axis=0)


def empty_metric(gt_line_count=0, pred_line_count=0):
    return {
        "gt_points_count": 0,
        "pred_points_count": 0,
        "gt_line_count": gt_line_count,
        "pred_line_count": pred_line_count,
        "gt_to_pred_mean_dist": None,
        "pred_to_gt_mean_dist": None,
        "recall_like": 0.0,
        "precision_like": 0.0,
        "f1_like": 0.0,
        "coverage_like": 0.0,
        "noise_penalty": 0.0,
        "has_gt": gt_line_count > 0,
    }


def score_class(gt_vectors, pred_vectors, args):
    gt_points = vectors_to_points(
        gt_vectors, args.sample_step, args.max_points_per_line
    )
    pred_points = vectors_to_points(
        pred_vectors, args.sample_step, args.max_points_per_line
    )
    metric = empty_metric(len(gt_vectors), len(pred_vectors))
    metric["gt_points_count"] = int(len(gt_points))
    metric["pred_points_count"] = int(len(pred_points))
    if len(gt_points) == 0:
        return metric
    metric["has_gt"] = True
    if len(pred_points) == 0:
        metric["noise_penalty"] = 0.0
        return metric

    gt_to_pred = nearest_distances(gt_points, pred_points)
    pred_to_gt = nearest_distances(pred_points, gt_points)
    recall = float(np.mean(gt_to_pred <= args.dist_thr))
    precision = float(np.mean(pred_to_gt <= args.dist_thr))
    f1 = float(2.0 * precision * recall / (precision + recall + 1e-8))
    far_pred = 1.0 - precision
    over_pred_ratio = max(0.0, len(pred_points) / max(float(len(gt_points)), 1.0) - 1.0)
    metric.update(
        {
            "gt_to_pred_mean_dist": float(np.mean(gt_to_pred)),
            "pred_to_gt_mean_dist": float(np.mean(pred_to_gt)),
            "recall_like": recall,
            "precision_like": precision,
            "f1_like": f1,
            "coverage_like": recall,
            "noise_penalty": float(far_pred + 0.1 * over_pred_ratio),
        }
    )
    return metric


def mean_metric(class_metrics, key, prefer_classes, min_gt_points):
    values = []
    for cls_name in prefer_classes:
        metric = class_metrics[cls_name]
        if metric["gt_points_count"] >= min_gt_points:
            values.append(metric[key])
    if not values:
        for cls_name in CLASS_NAMES:
            metric = class_metrics[cls_name]
            if metric["gt_points_count"] > 0:
                values.append(metric[key])
    return float(np.mean(values)) if values else 0.0


def score_sample(gt_item, pred_item, args, prefer_classes):
    gt_grouped = group_vectors(gt_item, score_thr=None)
    pred_grouped = group_vectors(pred_item, score_thr=args.score_thr)
    class_metrics = {}
    for cls_name in CLASS_NAMES:
        class_metrics[cls_name] = score_class(
            gt_grouped[cls_name], pred_grouped[cls_name], args
        )
    f1_mean = mean_metric(class_metrics, "f1_like", prefer_classes, args.min_gt_points)
    recall_mean = mean_metric(
        class_metrics, "recall_like", prefer_classes, args.min_gt_points
    )
    noise_mean = mean_metric(
        class_metrics, "noise_penalty", prefer_classes, args.min_gt_points
    )
    return {
        "classes": class_metrics,
        "f1_mean": f1_mean,
        "recall_mean": recall_mean,
        "noise_penalty_mean": noise_mean,
        "pred_points_total": int(
            sum(class_metrics[name]["pred_points_count"] for name in CLASS_NAMES)
        ),
    }


def metric_value(sample_scores, model, cls_name, key="f1_like"):
    return sample_scores[model]["classes"][cls_name][key]


def build_base_row(index, token, info, sample_scores, gt_scores):
    pose_f1 = sample_scores["pose"]["f1_mean"]
    dynamic_f1 = sample_scores["dynamic"]["f1_mean"]
    ego_f1 = sample_scores["ego"]["f1_mean"]
    row = {
        "sample_index": index,
        "sample_token": token,
        "scene_or_sample_name_if_available": info_name(info, token),
        "ego_f1_mean": sample_scores["ego"]["f1_mean"],
        "dynamic_f1_mean": dynamic_f1,
        "constant_f1_mean": sample_scores["constant"]["f1_mean"],
        "pose_f1_mean": pose_f1,
        "pose_minus_dynamic": pose_f1 - dynamic_f1,
        "pose_minus_ego": pose_f1 - ego_f1,
        "pose_noise_penalty": sample_scores["pose"]["noise_penalty_mean"],
        "gt_eval_points": gt_scores["gt_eval_points"],
        "gt_total_lines": gt_scores["gt_total_lines"],
        "pose_pred_points_total": sample_scores["pose"]["pred_points_total"],
        "dynamic_pred_points_total": sample_scores["dynamic"]["pred_points_total"],
        "recommended_visualization_command": (
            "bash scripts/visualize_maptr_bev_comparisons.sh "
            "--paper-4panel --overlay-gt --sample-indices {}".format(index)
        ),
    }
    for cls_name in CLASS_NAMES:
        row[f"gt_{cls_name}_count"] = gt_scores["class_line_counts"][cls_name]
        for model in ("ego", "dynamic", "constant", "pose"):
            row[f"{model}_{cls_name}_f1"] = metric_value(
                sample_scores, model, cls_name
            )
            row[f"{model}_{cls_name}_recall"] = metric_value(
                sample_scores, model, cls_name, "recall_like"
            )
            row[f"{model}_{cls_name}_pred_points"] = int(
                metric_value(sample_scores, model, cls_name, "pred_points_count")
            )
    return row


def compute_gt_summary(gt_item, args, prefer_classes):
    grouped = group_vectors(gt_item, score_thr=None)
    point_counts = {}
    line_counts = {}
    for cls_name in CLASS_NAMES:
        points = vectors_to_points(
            grouped[cls_name], args.sample_step, args.max_points_per_line
        )
        point_counts[cls_name] = int(len(points))
        line_counts[cls_name] = len(grouped[cls_name])
    return {
        "class_point_counts": point_counts,
        "class_line_counts": line_counts,
        "gt_eval_points": sum(point_counts[name] for name in prefer_classes),
        "gt_total_lines": sum(line_counts.values()),
    }


def finite_for_csv(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return value


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_columns = [
        "pose_noise_penalty",
        "gt_eval_points",
        "gt_total_lines",
        "pose_pred_points_total",
        "dynamic_pred_points_total",
    ]
    fieldnames = CSV_COLUMNS + [col for col in extra_columns if col not in CSV_COLUMNS]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: finite_for_csv(row.get(key, "")) for key in fieldnames})


def md_table(path, title, rows):
    indices = [str(row["sample_index"]) for row in rows]
    command = (
        "bash scripts/visualize_maptr_bev_comparisons.sh "
        "--paper-4panel --overlay-gt --sample-indices "
        + " ".join(indices)
        if indices
        else "N/A"
    )
    lines = [
        f"# {title}",
        "",
        "```bash",
        command,
        "```",
        "",
        "| rank | sample_index | score | pose_f1 | dynamic_f1 | ego_f1 | pose-dynamic | token |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {rank} | {sample_index} | {score:.4f} | {pose_f1_mean:.4f} | "
            "{dynamic_f1_mean:.4f} | {ego_f1_mean:.4f} | "
            "{pose_minus_dynamic:.4f} | `{sample_token}` |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n")


def rank_rows(rows, score_name, predicate, top_k):
    selected = []
    for row in rows:
        if predicate(row):
            copied = dict(row)
            copied["score"] = float(score_name(row))
            selected.append(copied)
    selected.sort(key=lambda item: item["score"], reverse=True)
    selected = selected[:top_k]
    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank
    return selected


def write_category(out_dir, filename_base, title, rows):
    csv_path = out_dir / f"{filename_base}.csv"
    md_path = out_dir / f"{filename_base}.md"
    json_path = out_dir / f"{filename_base}.json"
    write_csv(csv_path, rows)
    md_table(md_path, title, rows)
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    return csv_path, md_path, json_path


def category_rankings(rows, args):
    enough_gt = lambda row: row["gt_eval_points"] >= args.min_gt_points
    return {
        "pose_vs_dynamic": rank_rows(
            rows,
            lambda row: row["pose_minus_dynamic"],
            lambda row: (
                enough_gt(row)
                and row["pose_f1_mean"] > row["dynamic_f1_mean"]
                and row["pose_f1_mean"] > 0.05
            ),
            args.top_k,
        ),
        "pose_vs_ego": rank_rows(
            rows,
            lambda row: row["pose_minus_ego"],
            lambda row: enough_gt(row) and row["pose_f1_mean"] > row["ego_f1_mean"],
            args.top_k,
        ),
        "ped_crossing_recovery": rank_rows(
            rows,
            lambda row: row["pose_ped_crossing_f1"] - row["dynamic_ped_crossing_f1"],
            lambda row: (
                row["gt_ped_crossing_count"] > 0
                and row["pose_ped_crossing_f1"] > row["dynamic_ped_crossing_f1"]
                and row["pose_ped_crossing_pred_points"] >= 2
            ),
            args.top_k,
        ),
        "divider_recovery": rank_rows(
            rows,
            lambda row: row["pose_divider_f1"] - row["dynamic_divider_f1"],
            lambda row: (
                row["gt_divider_count"] > 0
                and row["pose_divider_f1"] > row["dynamic_divider_f1"]
            ),
            args.top_k,
        ),
        "clean_qualitative_candidates": rank_rows(
            rows,
            lambda row: (
                row["pose_minus_dynamic"]
                + 0.5 * row["pose_minus_ego"]
                + 0.3 * row["pose_ped_crossing_f1"]
                + 0.2 * row["pose_divider_f1"]
                - row["pose_noise_penalty"]
            ),
            lambda row: (
                enough_gt(row)
                and row["gt_total_lines"] >= 8
                and row["gt_total_lines"] <= 120
                and row["pose_f1_mean"] > row["dynamic_f1_mean"]
                and row["dynamic_pred_points_total"] > 0
                and row["pose_pred_points_total"] <= max(120, 4 * row["gt_eval_points"])
            ),
            args.top_k,
        ),
    }


def write_report(out_dir, args, matched_count, rankings, files):
    report = out_dir / "QUALITATIVE_RANKING_REPORT.md"
    lines = [
        "# Qualitative Ranking Report",
        "",
        "## Purpose",
        "",
        "Rank SimV2I-HD test samples where Pose-gated V2I is likely to be visually stronger than Dynamic Top-4 or Ego-only.",
        "",
        "## Inputs",
        "",
        f"- GT JSON: {args.gt_json}",
        f"- infos PKL: {args.infos}",
        f"- Ego-only prediction: {args.ego_pred}",
        f"- Dynamic Top-4 prediction: {args.dynamic_pred}",
        f"- Constant Gate prediction: {args.constant_pred}",
        f"- Pose-gated prediction: {args.pose_pred}",
        "",
        "## Matching",
        "",
        f"- Matched samples by sample_token: {matched_count}/4000",
        "- Ranking never assumes prediction list order equals GT order.",
        "",
        "## Settings",
        "",
        f"- score_thr: {args.score_thr}",
        f"- dist_thr: {args.dist_thr}",
        f"- top_k: {args.top_k}",
        f"- min_gt_points: {args.min_gt_points}",
        f"- prefer_classes: {args.prefer_classes}",
        "",
        "## Approximate Metric",
        "",
        "- Each polyline is uniformly sampled into an ego-BEV point set.",
        "- KDTree nearest-neighbor distances are computed GT-to-prediction and prediction-to-GT.",
        "- recall_like is the fraction of GT points within dist_thr of predictions.",
        "- precision_like is the fraction of predicted points within dist_thr of GT.",
        "- f1_like is the harmonic mean of precision_like and recall_like.",
        "- noise_penalty increases when predicted points are far from GT or greatly outnumber GT points.",
        "",
        "## Ranking Categories",
        "",
        "- pose_vs_dynamic: Pose-gated recovery over Dynamic Top-4.",
        "- pose_vs_ego: Pose-gated improvement over Ego-only.",
        "- ped_crossing_recovery: Pose-gated pedestrian crossing recovery over Dynamic Top-4.",
        "- divider_recovery: Pose-gated divider recovery over Dynamic Top-4.",
        "- clean_qualitative_candidates: combined score favoring readable, non-noisy paper candidates.",
        "",
        "## Category Top-10",
        "",
    ]
    for category, rows in rankings.items():
        lines.append(f"### {category}")
        if not rows:
            lines.append("")
            lines.append("- No candidates found.")
            lines.append("")
            continue
        lines.append("")
        lines.append("| rank | sample_index | score | token |")
        lines.append("|---:|---:|---:|---|")
        for row in rows[:10]:
            lines.append(
                f"| {row['rank']} | {row['sample_index']} | {row['score']:.4f} | `{row['sample_token']}` |"
            )
        lines.append("")

    lines.extend(
        [
            "## Output Files",
            "",
        ]
    )
    for file_path in files:
        lines.append(f"- {file_path}")
    lines.extend(
        [
            "",
            "## Example Visualization Command",
            "",
            "```bash",
            "bash scripts/visualize_maptr_bev_comparisons.sh --paper-4panel --overlay-gt --sample-indices "
            + " ".join(
                str(row["sample_index"])
                for row in rankings.get("clean_qualitative_candidates", [])[:10]
            ),
            "```",
            "",
            "## Limitations",
            "",
            "- This ranking score is an approximate metric for qualitative candidate selection, not the official paper quantitative metric.",
            "- Final paper figures should be manually checked and selected as representative cases.",
            "- Result interpretation should be based on the official Section 4 NuscMap_chamfer/mAP numbers.",
        ]
    )
    report.write_text("\n".join(lines) + "\n")
    return report


def main():
    args = parse_args()
    prefer_classes = parse_classes(args.prefer_classes)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, gt_records, _ = load_gt_records(args.gt_json)
    infos_obj, infos, info_tokens, _ = load_info_tokens(args.infos)
    info_by_token = {str(info.get("token")): info for info in infos}

    pred_paths = {
        "ego": args.ego_pred,
        "dynamic": args.dynamic_pred,
        "constant": args.constant_pred,
        "pose": args.pose_pred,
    }
    predictions = {}
    for model, path in pred_paths.items():
        _, _, token_to_pred, _ = load_prediction_records(path, gt_records)
        predictions[model] = token_to_pred

    gt_tokens = [str(item.get("sample_token") or item.get("token")) for item in gt_records]
    matched_count = sum(
        1 for token in gt_tokens if all(token in pred for pred in predictions.values())
    )

    rows = []
    for index, gt_item in enumerate(gt_records):
        token = str(gt_item.get("sample_token") or gt_item.get("token") or index)
        sample_scores = {}
        for model, token_to_pred in predictions.items():
            sample_scores[model] = score_sample(
                gt_item, token_to_pred.get(token, {"vectors": []}), args, prefer_classes
            )
        gt_summary = compute_gt_summary(gt_item, args, prefer_classes)
        rows.append(
            build_base_row(
                index,
                token,
                info_by_token.get(token),
                sample_scores,
                gt_summary,
            )
        )

    rankings = category_rankings(rows, args)
    files = []
    category_titles = {
        "pose_vs_dynamic": "Pose-gated vs Dynamic Top-4 Recovery",
        "pose_vs_ego": "Pose-gated vs Ego-only Improvement",
        "ped_crossing_recovery": "Pedestrian Crossing Recovery",
        "divider_recovery": "Divider Recovery",
        "clean_qualitative_candidates": "Clean Qualitative Candidates",
    }
    for category, ranked_rows in rankings.items():
        files.extend(
            write_category(out_dir, category, category_titles[category], ranked_rows)
        )

    all_scores_path = out_dir / "all_sample_scores.json"
    all_scores_path.write_text(json.dumps(rows, indent=2) + "\n")
    files.append(all_scores_path)
    top_indices_path = out_dir / "top_indices_by_category.json"
    top_indices = {
        category: [row["sample_index"] for row in ranked_rows]
        for category, ranked_rows in rankings.items()
    }
    top_indices_path.write_text(json.dumps(top_indices, indent=2) + "\n")
    files.append(top_indices_path)

    report = write_report(out_dir, args, matched_count, rankings, files)

    print(f"Matched samples by sample_token: {matched_count}/{len(gt_records)}")
    for category, ranked_rows in rankings.items():
        indices = [str(row["sample_index"]) for row in ranked_rows[:10]]
        print(f"{category} top indices: {' '.join(indices) if indices else 'NONE'}")
    print(f"Report: {report}")
    print(
        "Example command: bash scripts/visualize_maptr_bev_comparisons.sh "
        "--paper-4panel --overlay-gt --sample-indices "
        + " ".join(
            str(row["sample_index"])
            for row in rankings.get("clean_qualitative_candidates", [])[:10]
        )
    )


if __name__ == "__main__":
    main()
