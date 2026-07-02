#!/usr/bin/env python
"""Plot qualitative SimV2I-HD MapTR comparison cases for supplementary figures.

The default workflow uses saved prediction files and does not run inference:

    python tools/plot_qualitative_cases.py \
      --data-root "${SIMV2I_HD_ROOT}" \
      --split test \
      --ego-pred outputs/maptr/visualization_predictions/ego_only_test_epoch18.pkl \
      --dynamic-top4-pred outputs/maptr/visualization_predictions/dynamic_top4_test_epoch18.pkl \
      --pose-gated-pred outputs/maptr/visualization_predictions/pose_gated_top4_test_epoch18.pkl \
      --output-dir outputs/qualitative_cases \
      --num-cases 5

If a prediction file is missing, the script can optionally run evaluation-only
inference with --run-missing-inference and the corresponding config/checkpoint
arguments. No training logic is touched.
"""

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-maptr")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/maptr-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rank_qualitative_pose_gated_samples import (  # noqa: E402
    CLASS_NAMES,
    compute_gt_summary,
    group_vectors,
    info_name,
    parse_classes,
    score_sample,
)
from visualize_simv2i_maptr_predictions import (  # noqa: E402
    DISPLAY_NAMES,
    PAPER_READY_STYLE,
    apply_text_style,
    describe_structure,
    load_gt_records,
    load_info_tokens,
    load_prediction_records,
    make_figure,
)


DEFAULT_DATA_ROOT = os.environ.get(
    "SIMV2I_HD_ROOT",
    "data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k",
)
DEFAULT_PREDS = {
    "ego_only": "outputs/maptr/visualization_predictions/ego_only_test_epoch18.pkl",
    "dynamic_top4": (
        "outputs/maptr/visualization_predictions/dynamic_top4_test_epoch18.pkl"
    ),
    "pose_gated_top4": (
        "outputs/maptr/visualization_predictions/pose_gated_top4_test_epoch18.pkl"
    ),
}
MODEL_LABELS = {
    "ego_only": "Ego-only",
    "dynamic_top4": "Dynamic Top-4",
    "pose_gated_top4": "Pose-gated Top-4",
}
DISPLAY_NAMES["pose_gated_top4"] = "Pose-gated Top-4"
CSV_COLUMNS = [
    "case_rank",
    "category",
    "selection_reason",
    "sample_index",
    "sample_id",
    "sample_token",
    "scene_or_sample_name_if_available",
    "ego_f1_mean",
    "dynamic_top4_f1_mean",
    "pose_gated_f1_mean",
    "pose_minus_ego",
    "pose_minus_dynamic",
    "ego_minus_dynamic",
    "gt_total_lines",
    "gt_divider_count",
    "gt_boundary_count",
    "gt_ped_crossing_count",
    "png_path",
    "pdf_path",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate GT / Ego-only / Dynamic Top-4 / Pose-gated Top-4 "
            "BEV qualitative comparison figures."
        )
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--gt-json", help="Override GT vector-map JSON path.")
    parser.add_argument("--infos-pkl", help="Override infos PKL path.")
    parser.add_argument("--ego-pred", default=DEFAULT_PREDS["ego_only"])
    parser.add_argument("--dynamic-top4-pred", default=DEFAULT_PREDS["dynamic_top4"])
    parser.add_argument("--pose-gated-pred", default=DEFAULT_PREDS["pose_gated_top4"])
    parser.add_argument("--output-dir", default="outputs/qualitative_cases")
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        help=(
            "Optional sample indices or sample tokens. When provided, these "
            "manual cases are plotted in the given order."
        ),
    )
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--dist-thr", type=float, default=1.5)
    parser.add_argument("--min-gt-points", type=int, default=20)
    parser.add_argument("--prefer-classes", default="divider,ped_crossing")
    parser.add_argument("--sample-step", type=float, default=1.0)
    parser.add_argument("--max-points-per-line", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--selection-mode",
        choices=("auto", "random"),
        default="auto",
        help="Auto uses approximate per-sample metrics; random is illustrative.",
    )
    parser.add_argument(
        "--overlay-gt",
        action="store_true",
        help="Overlay faint GT lines on prediction panels.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Only save PNG files.",
    )
    parser.add_argument(
        "--map-range",
        type=float,
        nargs=4,
        default=(-15.0, -30.0, 15.0, 30.0),
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
    )
    parser.add_argument("--fig-width", type=float, default=11.2)
    parser.add_argument("--fig-height", type=float, default=4.35)
    parser.add_argument("--wspace", type=float, default=0.045)
    parser.add_argument("--title-fontsize", type=float, default=11.0)
    parser.add_argument("--panel-title-fontsize", type=float, default=10.5)
    parser.add_argument("--axis-fontsize", type=float, default=8.5)
    parser.add_argument("--tick-fontsize", type=float, default=8.0)
    parser.add_argument("--legend-fontsize", type=float, default=8.0)

    parser.add_argument("--run-missing-inference", action="store_true")
    parser.add_argument("--ego-config")
    parser.add_argument("--ego-checkpoint")
    parser.add_argument("--dynamic-config")
    parser.add_argument("--dynamic-checkpoint")
    parser.add_argument("--pose-config")
    parser.add_argument("--pose-checkpoint")
    parser.add_argument("--eval-gpus", type=int, default=1)
    parser.add_argument("--eval-port", type=int, default=29917)
    return parser.parse_args()


def split_paths(args):
    data_root = Path(args.data_root)
    gt_json = Path(args.gt_json) if args.gt_json else data_root / (
        "simv2i_maptr_map_gt_{}.json".format(args.split)
    )
    infos_pkl = Path(args.infos_pkl) if args.infos_pkl else data_root / (
        "simv2i_maptr_infos_{}.pkl".format(args.split)
    )
    return gt_json, infos_pkl


def metric_args(args):
    return SimpleNamespace(
        score_thr=args.score_thr,
        dist_thr=args.dist_thr,
        min_gt_points=args.min_gt_points,
        sample_step=args.sample_step,
        max_points_per_line=args.max_points_per_line,
    )


def run_eval_inference(model_name, config, checkpoint, output_dir, gpus, port):
    if not config or not checkpoint:
        raise FileNotFoundError(
            "Missing prediction for {} and no config/checkpoint was provided.".format(
                model_name
            )
        )
    result_prefix = output_dir / "inference" / model_name / "results"
    result_json = result_prefix / "pts_bbox" / "nuscmap_results.json"
    if result_json.exists():
        return result_json

    command = [
        sys.executable,
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node={}".format(gpus),
        "--master_port={}".format(port),
        "tools/test.py",
        config,
        checkpoint,
        "--launcher",
        "pytorch",
        "--eval",
        "chamfer",
        "--eval-options",
        "jsonfile_prefix={}".format(result_prefix),
    ]
    print("Running evaluation-only inference for {}:".format(model_name))
    print(" ".join(command))
    subprocess.run(command, check=True)
    if not result_json.exists():
        raise FileNotFoundError(
            "Evaluation finished but result JSON was not found: {}".format(result_json)
        )
    return result_json


def resolve_prediction_path(model_name, pred_path, args, output_dir):
    path = Path(pred_path)
    if path.exists():
        return path
    if not args.run_missing_inference:
        raise FileNotFoundError(
            "Prediction file not found for {}: {}\n"
            "Provide an existing prediction file or rerun with "
            "--run-missing-inference and the matching config/checkpoint.".format(
                model_name, path
            )
        )
    if model_name == "ego_only":
        return run_eval_inference(
            model_name,
            args.ego_config,
            args.ego_checkpoint,
            output_dir,
            args.eval_gpus,
            args.eval_port,
        )
    if model_name == "dynamic_top4":
        return run_eval_inference(
            model_name,
            args.dynamic_config,
            args.dynamic_checkpoint,
            output_dir,
            args.eval_gpus,
            args.eval_port + 1,
        )
    return run_eval_inference(
        model_name,
        args.pose_config,
        args.pose_checkpoint,
        output_dir,
        args.eval_gpus,
        args.eval_port + 2,
    )


def safe_float(value):
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(value) or math.isinf(value):
        return ""
    return "{:.6f}".format(value)


def count_gt_lines(gt_item):
    grouped = group_vectors(gt_item, score_thr=None)
    return {cls_name: len(grouped[cls_name]) for cls_name in CLASS_NAMES}


def build_score_rows(gt_records, infos, predictions, args):
    prefer_classes = parse_classes(args.prefer_classes)
    scorer_args = metric_args(args)
    info_by_token = {str(info.get("token")): info for info in infos}
    rows = []
    for index, gt_item in enumerate(gt_records):
        token = str(gt_item.get("sample_token") or gt_item.get("token") or index)
        sample_scores = {}
        for model_name, token_to_pred in predictions.items():
            sample_scores[model_name] = score_sample(
                gt_item,
                token_to_pred.get(token, {"vectors": []}),
                scorer_args,
                prefer_classes,
            )
        gt_summary = compute_gt_summary(gt_item, scorer_args, prefer_classes)
        line_counts = count_gt_lines(gt_item)
        pose_f1 = sample_scores["pose_gated_top4"]["f1_mean"]
        dynamic_f1 = sample_scores["dynamic_top4"]["f1_mean"]
        ego_f1 = sample_scores["ego_only"]["f1_mean"]
        rows.append(
            {
                "sample_index": index,
                "sample_id": str(index),
                "sample_token": token,
                "scene_or_sample_name_if_available": info_name(
                    info_by_token.get(token), token
                ),
                "ego_f1_mean": ego_f1,
                "dynamic_top4_f1_mean": dynamic_f1,
                "pose_gated_f1_mean": pose_f1,
                "pose_minus_ego": pose_f1 - ego_f1,
                "pose_minus_dynamic": pose_f1 - dynamic_f1,
                "ego_minus_dynamic": ego_f1 - dynamic_f1,
                "gt_eval_points": gt_summary["gt_eval_points"],
                "gt_total_lines": gt_summary["gt_total_lines"],
                "gt_divider_count": line_counts["divider"],
                "gt_boundary_count": line_counts["boundary"],
                "gt_ped_crossing_count": line_counts["ped_crossing"],
            }
        )
        if (index + 1) % 500 == 0:
            print("Scored {}/{} samples".format(index + 1, len(gt_records)))
    return rows


def sample_id_to_index(sample_id, gt_records, token_to_index):
    if str(sample_id).isdigit():
        index = int(sample_id)
        if 0 <= index < len(gt_records):
            return index
    if sample_id in token_to_index:
        return token_to_index[sample_id]
    raise ValueError("Unknown sample id/token: {}".format(sample_id))


def random_cases(gt_records, args):
    rng = random.Random(args.seed)
    indices = list(range(len(gt_records)))
    rng.shuffle(indices)
    selected = []
    for index in indices[: args.num_cases]:
        gt_item = gt_records[index]
        token = str(gt_item.get("sample_token") or gt_item.get("token") or index)
        selected.append(
            {
                "case_rank": len(selected) + 1,
                "category": "illustrative_random",
                "selection_reason": (
                    "Random illustrative example; no per-sample metric category "
                    "was used."
                ),
                "sample_index": index,
                "sample_id": str(index),
                "sample_token": token,
            }
        )
    return selected


def auto_select_cases(rows, args):
    enough_gt = lambda row: row["gt_eval_points"] >= args.min_gt_points
    selected = []
    used = set()

    def take(category, reason, candidates, count):
        nonlocal selected
        for row in candidates:
            index = row["sample_index"]
            if index in used:
                continue
            item = dict(row)
            item["case_rank"] = len(selected) + 1
            item["category"] = category
            item["selection_reason"] = reason
            selected.append(item)
            used.add(index)
            if sum(1 for case in selected if case["category"] == category) >= count:
                break

    pose_over_ego = sorted(
        [
            row
            for row in rows
            if enough_gt(row)
            and row["pose_gated_f1_mean"] > row["ego_f1_mean"]
            and row["pose_gated_f1_mean"] > 0.05
        ],
        key=lambda row: row["pose_minus_ego"],
        reverse=True,
    )
    take(
        "pose_improves_over_ego",
        "Pose-gated F1-like score improves over Ego-only by a large margin.",
        pose_over_ego,
        2,
    )

    dynamic_worse = sorted(
        [
            row
            for row in rows
            if enough_gt(row)
            and row["dynamic_top4_f1_mean"] < row["ego_f1_mean"]
        ],
        key=lambda row: row["ego_minus_dynamic"],
        reverse=True,
    )
    take(
        "dynamic_top4_worse_than_ego",
        "Dynamic Top-4 F1-like score is lower than Ego-only.",
        dynamic_worse,
        2,
    )

    pose_failure = sorted(
        [row for row in rows if enough_gt(row)],
        key=lambda row: (
            row["pose_gated_f1_mean"],
            -max(row["ego_f1_mean"], row["dynamic_top4_f1_mean"]),
        ),
    )
    take(
        "pose_gated_failure",
        "Pose-gated has a low F1-like score; useful as an illustrative failure case.",
        pose_failure,
        1,
    )

    if len(selected) < args.num_cases:
        extras = sorted(
            [row for row in rows if enough_gt(row)],
            key=lambda row: (
                row["pose_gated_f1_mean"]
                + row["pose_minus_dynamic"]
                + 0.5 * row["pose_minus_ego"]
            ),
            reverse=True,
        )
        take(
            "supplementary_extra",
            "Additional high-scoring illustrative supplementary example.",
            extras,
            args.num_cases - len(selected),
        )

    selected = selected[: args.num_cases]
    for rank, item in enumerate(selected, start=1):
        item["case_rank"] = rank
    return selected


def manual_cases(sample_ids, gt_records, rows_by_index):
    token_to_index = {
        str(item.get("sample_token") or item.get("token") or index): index
        for index, item in enumerate(gt_records)
    }
    selected = []
    for sample_id in sample_ids:
        index = sample_id_to_index(sample_id, gt_records, token_to_index)
        gt_item = gt_records[index]
        token = str(gt_item.get("sample_token") or gt_item.get("token") or index)
        item = dict(rows_by_index.get(index, {}))
        item.update(
            {
                "case_rank": len(selected) + 1,
                "category": "manual",
                "selection_reason": "Manually requested via --sample-ids.",
                "sample_index": index,
                "sample_id": str(index),
                "sample_token": token,
            }
        )
        selected.append(item)
    return selected


def build_style(args):
    style = dict(PAPER_READY_STYLE)
    style.update(
        {
            "fig_width": args.fig_width,
            "fig_height": args.fig_height,
            "wspace": args.wspace,
            "subplot_left": 0.055,
            "subplot_right": 0.995,
            "subplot_top": 0.82,
            "subplot_bottom": 0.205,
            "title_fontsize": args.title_fontsize,
            "panel_title_fontsize": args.panel_title_fontsize,
            "axis_fontsize": args.axis_fontsize,
            "tick_fontsize": args.tick_fontsize,
            "legend_fontsize": args.legend_fontsize,
            "font_family": "serif",
            "pred_linewidth": 1.75,
            "gt_panel_linewidth": 1.05,
            "gt_overlay_linewidth": 0.8,
            "gt_overlay_alpha": 0.22,
        }
    )
    return style


def write_selected_csv(path, selected):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    key: safe_float(row.get(key, ""))
                    if key.endswith("_mean")
                    or key.startswith("pose_minus")
                    or key == "ego_minus_dynamic"
                    else row.get(key, "")
                    for key in CSV_COLUMNS
                }
            )


def write_report(path, args, gt_json, infos_pkl, pred_paths, pred_structures, selected):
    lines = [
        "# Qualitative BEV Vector Map Cases",
        "",
        "These are illustrative supplementary examples generated from existing saved predictions/checkpoints. No model training was started by this script.",
        "",
        "## Inputs",
        "",
        "- Split: `{}`".format(args.split),
        "- GT JSON: `{}`".format(gt_json),
        "- infos PKL: `{}`".format(infos_pkl),
    ]
    for model_name, pred_path in pred_paths.items():
        lines.append("- {} prediction: `{}`".format(MODEL_LABELS[model_name], pred_path))
    lines.extend(["", "## Prediction Structures", ""])
    for model_name, structure in pred_structures.items():
        lines.append("- {}: {}".format(MODEL_LABELS[model_name], structure))
    lines.extend(
        [
            "",
            "## Plot Settings",
            "",
            "- Axis range: x=[{}, {}], y=[{}, {}]".format(
                args.map_range[0],
                args.map_range[2],
                args.map_range[1],
                args.map_range[3],
            ),
            "- Score threshold: `{}`".format(args.score_thr),
            "- Approximate metric distance threshold: `{}`".format(args.dist_thr),
            "- Selection mode: `{}`".format(
                "manual" if args.sample_ids else args.selection_mode
            ),
            "",
            "## Selected Cases",
            "",
            "| rank | category | sample_index | ego | dynamic | pose | reason | figure |",
            "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in selected:
        lines.append(
            "| {rank} | {category} | {index} | {ego} | {dynamic} | {pose} | {reason} | `{figure}` |".format(
                rank=row["case_rank"],
                category=row["category"],
                index=row["sample_index"],
                ego=safe_float(row.get("ego_f1_mean")),
                dynamic=safe_float(row.get("dynamic_top4_f1_mean")),
                pose=safe_float(row.get("pose_gated_f1_mean")),
                reason=row.get("selection_reason", ""),
                figure=row.get("png_path", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The automatic categories use an approximate sampled-polyline F1-like score for qualitative case selection, not the official AP/Chamfer metric.",
            "- The figures compare GT, Ego-only, Dynamic Top-4, and Pose-gated Top-4 on the same test samples.",
            "- Interpret these as illustrative examples; final paper claims should rely on official quantitative results.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_json, infos_pkl = split_paths(args)
    if not gt_json.exists():
        raise FileNotFoundError("GT JSON not found: {}".format(gt_json))
    if not infos_pkl.exists():
        raise FileNotFoundError("infos PKL not found: {}".format(infos_pkl))

    pred_paths = {
        "ego_only": resolve_prediction_path(
            "ego_only", args.ego_pred, args, output_dir
        ),
        "dynamic_top4": resolve_prediction_path(
            "dynamic_top4", args.dynamic_top4_pred, args, output_dir
        ),
        "pose_gated_top4": resolve_prediction_path(
            "pose_gated_top4", args.pose_gated_pred, args, output_dir
        ),
    }

    gt_obj, gt_records, _ = load_gt_records(gt_json)
    infos_obj, infos, _, _ = load_info_tokens(infos_pkl)
    predictions = {}
    pred_structures = {}
    for model_name, pred_path in pred_paths.items():
        _, _, token_to_pred, structure = load_prediction_records(pred_path, gt_records)
        predictions[model_name] = token_to_pred
        pred_structures[model_name] = structure

    gt_tokens = [
        str(item.get("sample_token") or item.get("token") or index)
        for index, item in enumerate(gt_records)
    ]
    matched = {
        model_name: sum(token in token_to_pred for token in gt_tokens)
        for model_name, token_to_pred in predictions.items()
    }
    print("GT structure: {}".format(describe_structure(gt_obj)))
    print("infos structure: {}".format(describe_structure(infos_obj)))
    for model_name, count in matched.items():
        print("{} matched {}/{} samples".format(model_name, count, len(gt_records)))

    score_rows = []
    if args.selection_mode == "auto" or args.sample_ids:
        score_rows = build_score_rows(gt_records, infos, predictions, args)
    rows_by_index = {row["sample_index"]: row for row in score_rows}

    if args.sample_ids:
        selected = manual_cases(args.sample_ids, gt_records, rows_by_index)
        if args.num_cases:
            selected = selected[: args.num_cases]
    elif args.selection_mode == "random":
        selected = random_cases(gt_records, args)
    else:
        selected = auto_select_cases(score_rows, args)
        if not selected:
            print("No metric-selected cases found; falling back to random examples.")
            selected = random_cases(gt_records, args)

    style = build_style(args)
    apply_text_style(style)
    model_order = ["ego_only", "dynamic_top4", "pose_gated_top4"]
    pred_items = predictions

    for row in selected:
        index = int(row["sample_index"])
        gt_item = gt_records[index]
        token = str(gt_item.get("sample_token") or gt_item.get("token") or index)
        title = "Sample ID {}".format(row.get("sample_id", index))
        fig = make_figure(
            index,
            token,
            gt_item,
            pred_items,
            model_order,
            list(args.map_range),
            args.score_thr,
            overlay_gt=args.overlay_gt,
            style=style,
            title=title,
            hide_token=True,
        )
        out_base = output_dir / "qualitative_case_{:04d}".format(index)
        png_path = out_base.with_suffix(".png")
        pdf_path = out_base.with_suffix(".pdf")
        fig.savefig(png_path, dpi=300)
        if not args.no_pdf:
            fig.savefig(pdf_path)
            row["pdf_path"] = str(pdf_path)
        else:
            row["pdf_path"] = ""
        plt.close(fig)
        row["png_path"] = str(png_path)

    selected_csv = output_dir / "selected_cases.csv"
    report_path = output_dir / "qualitative_cases.md"
    write_selected_csv(selected_csv, selected)
    write_report(
        report_path,
        args,
        gt_json,
        infos_pkl,
        pred_paths,
        pred_structures,
        selected,
    )

    print("Wrote {}".format(selected_csv))
    print("Wrote {}".format(report_path))
    for row in selected:
        print("Wrote {}".format(row["png_path"]))
        if row.get("pdf_path"):
            print("Wrote {}".format(row["pdf_path"]))


if __name__ == "__main__":
    main()
