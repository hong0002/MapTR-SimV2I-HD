#!/usr/bin/env python
"""Visualize SimV2I-HD MapTR vector-map predictions in ego BEV."""

import argparse
import ast
import json
import os
import pickle
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-maptr")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/maptr-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np


CLASS_NAMES = ("divider", "boundary", "ped_crossing")
SERIF_FONT_FALLBACKS = (
    "Times New Roman",
    "Times",
    "Nimbus Roman",
    "Nimbus Roman No9 L",
    "Liberation Serif",
    "DejaVu Serif",
)
DISPLAY_NAMES = {
    "ego_only": "Ego-only",
    "dynamic_top4": "Dynamic Top-4",
    "constant_gate": "Constant Gate",
    "pose_gated_top4": "Pose-gated V2I",
}
MODEL_SHORT_NAMES = {
    "ego_only": "ego",
    "dynamic_top4": "dynamic",
    "constant_gate": "constant",
    "pose_gated_top4": "pose",
}
CHECKPOINTS = {
    "ego_only": "outputs/maptr/ego_only_r18_20k_b4/epoch_18.pth",
    "dynamic_top4": "outputs/maptr/v2_dynamic_rsu_top4_r18_20k/epoch_18.pth",
    "constant_gate": "outputs/maptr/gate_constant_v2i_r18_20k_b4/epoch_18.pth",
    "pose_gated_top4": "outputs/maptr/pose_gated_v2i_r18_20k_b4/epoch_18.pth",
}
PRED_COLORS = {
    "divider": "#1f77b4",
    "boundary": "#d55e00",
    "ped_crossing": "#009e73",
}
GT_STYLES = {
    "divider": ("#111111", "-"),
    "boundary": ("#333333", "--"),
    "ped_crossing": ("#555555", ":"),
}
PRED_STYLES = {
    "divider": "-",
    "boundary": "--",
    "ped_crossing": "-.",
}
DEFAULT_STYLE = {
    "gt_overlay_alpha": 0.35,
    "gt_overlay_linewidth": 0.8,
    "gt_panel_alpha": 0.9,
    "gt_panel_linewidth": 1.2,
    "pred_alpha": 0.95,
    "pred_linewidth": 1.3,
    "legend_fontsize": 8,
    "title_fontsize": 11,
    "panel_title_fontsize": 10,
    "axis_fontsize": 8,
    "tick_fontsize": 8,
    "font_family": None,
    "font_serif": SERIF_FONT_FALLBACKS,
    "grid_linewidth": 0.4,
    "grid_alpha": 1.0,
    "fig_width_per_panel": 4.0,
    "fig_height": 6.2,
    "wspace": None,
    "subplot_left": 0.0,
    "subplot_right": 1.0,
    "subplot_top": 0.95,
    "subplot_bottom": 0.055,
}
PAPER_READY_STYLE = {
    "gt_overlay_alpha": 0.28,
    "gt_overlay_linewidth": 1.0,
    "gt_panel_alpha": 0.92,
    "gt_panel_linewidth": 1.15,
    "pred_alpha": 1.0,
    "pred_linewidth": 2.0,
    "legend_fontsize": 9,
    "title_fontsize": 13,
    "panel_title_fontsize": 12,
    "axis_fontsize": 10,
    "tick_fontsize": 9.5,
    "font_family": "serif",
    "font_serif": SERIF_FONT_FALLBACKS,
    "grid_linewidth": 0.35,
    "grid_alpha": 0.65,
    "fig_width_per_panel": 2.6,
    "fig_height": 4.2,
    "wspace": 0.035,
    "subplot_left": 0.045,
    "subplot_right": 0.995,
    "subplot_top": 0.86,
    "subplot_bottom": 0.205,
}

DEFAULT_DATA_ROOT = os.environ.get(
    "SIMV2I_HD_ROOT",
    "data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw GT and MapTR predictions as BEV vector maps."
    )
    parser.add_argument(
        "--gt-json",
        default=os.path.join(DEFAULT_DATA_ROOT, "simv2i_maptr_map_gt_test.json"),
    )
    parser.add_argument(
        "--infos-pkl",
        default=os.path.join(DEFAULT_DATA_ROOT, "simv2i_maptr_infos_test.pkl"),
    )
    parser.add_argument(
        "--pred-pkl",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help="Prediction pkl/json. May be passed multiple times.",
    )
    parser.add_argument(
        "--config",
        default="integrations/maptr/configs/simv2i_maptr_ego_only_r18_20k_b4.py",
        help="Config used to infer pc_range when --map-range is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/maptr/visualizations/bev_map_comparison",
    )
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--sample-indices", type=int, nargs="+")
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument(
        "--overlay-gt",
        action="store_true",
        help="Overlay GT lines on each prediction panel.",
    )
    parser.add_argument(
        "--paper-4panel",
        action="store_true",
        help="Use GT / Ego-only / Dynamic Top-4 / Pose-gated V2I panels.",
    )
    parser.add_argument(
        "--paper-5panel",
        action="store_true",
        help="Use GT / Ego-only / Dynamic Top-4 / Constant Gate / Pose-gated V2I.",
    )
    parser.add_argument(
        "--filename-prefix",
        default="",
        help="Optional prefix added to each generated PNG filename.",
    )
    parser.add_argument(
        "--output-name",
        help=(
            "Optional exact basename without extension. Best used with a "
            "single sample index."
        ),
    )
    parser.add_argument(
        "--paper-ready",
        action="store_true",
        help="Apply compact paper styling without changing default behavior.",
    )
    parser.add_argument(
        "--short-title",
        help="Figure title. With --paper-ready defaults to 'Sample <index>'.",
    )
    parser.add_argument(
        "--hide-token",
        action="store_true",
        help="Hide the long sample token in the figure title.",
    )
    parser.add_argument("--gt-alpha", type=float)
    parser.add_argument("--gt-linewidth", type=float)
    parser.add_argument("--pred-linewidth", type=float)
    parser.add_argument("--legend-fontsize", type=float)
    parser.add_argument("--title-fontsize", type=float)
    parser.add_argument("--panel-title-fontsize", type=float)
    parser.add_argument("--axis-fontsize", type=float)
    parser.add_argument("--tick-fontsize", type=float)
    parser.add_argument("--wspace", type=float)
    parser.add_argument("--fig-width", type=float)
    parser.add_argument("--fig-height", type=float)
    parser.add_argument(
        "--font-family",
        help=(
            "Preferred font family. With --paper-ready, defaults to Times New "
            "Roman with serif fallbacks."
        ),
    )
    parser.add_argument(
        "--save-pdf",
        action="store_true",
        help="Save a PDF next to the PNG.",
    )
    parser.add_argument(
        "--map-range",
        type=float,
        nargs=4,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="Override BEV axis range. Defaults to config pc_range.",
    )
    parser.add_argument(
        "--report-path",
        default="outputs/maptr/visualizations/VISUALIZATION_REPORT.md",
    )
    args = parser.parse_args()
    if args.paper_4panel and args.paper_5panel:
        parser.error("--paper-4panel and --paper-5panel cannot be used together")
    return args


def choose_serif_font(preferred=None):
    installed = {font.name for font in font_manager.fontManager.ttflist}
    candidates = (preferred,) + SERIF_FONT_FALLBACKS if preferred else SERIF_FONT_FALLBACKS
    for candidate in candidates:
        if candidate and candidate in installed:
            return candidate
    return "DejaVu Serif"


def resolve_style(args, panel_count):
    style = dict(PAPER_READY_STYLE if args.paper_ready else DEFAULT_STYLE)
    if args.paper_ready:
        chosen_font = choose_serif_font(args.font_family)
        style["font_family"] = chosen_font
        style["font_serif"] = (chosen_font,) + tuple(
            name for name in SERIF_FONT_FALLBACKS if name != chosen_font
        )
    elif args.font_family:
        style["font_family"] = args.font_family
    if args.gt_alpha is not None:
        style["gt_overlay_alpha"] = args.gt_alpha
    if args.gt_linewidth is not None:
        style["gt_overlay_linewidth"] = args.gt_linewidth
        if args.paper_ready:
            style["gt_panel_linewidth"] = max(args.gt_linewidth, 1.0)
    if args.pred_linewidth is not None:
        style["pred_linewidth"] = args.pred_linewidth
    if args.legend_fontsize is not None:
        style["legend_fontsize"] = args.legend_fontsize
    if args.title_fontsize is not None:
        style["title_fontsize"] = args.title_fontsize
    if args.panel_title_fontsize is not None:
        style["panel_title_fontsize"] = args.panel_title_fontsize
    if args.axis_fontsize is not None:
        style["axis_fontsize"] = args.axis_fontsize
    if args.tick_fontsize is not None:
        style["tick_fontsize"] = args.tick_fontsize
    if args.wspace is not None:
        style["wspace"] = args.wspace
    if args.fig_height is not None:
        style["fig_height"] = args.fig_height
    if args.fig_width is not None:
        style["fig_width"] = args.fig_width
    else:
        style["fig_width"] = max(style["fig_width_per_panel"] * panel_count, 8.0)
    return style


def apply_text_style(style):
    font_family = style.get("font_family")
    if not font_family:
        return
    if font_family == "serif":
        plt.rcParams["font.family"] = "serif"
    else:
        plt.rcParams["font.family"] = font_family
    plt.rcParams["font.serif"] = list(style.get("font_serif", SERIF_FONT_FALLBACKS))
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_any(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        return load_json(path)
    return load_pickle(path)


def parse_pred_specs(specs):
    if not specs:
        specs = [
            "ego_only=outputs/maptr/visualization_predictions/"
            "ego_only_test_epoch18.pkl",
            "dynamic_top4=outputs/maptr/visualization_predictions/"
            "dynamic_top4_test_epoch18.pkl",
            "constant_gate=outputs/maptr/visualization_predictions/"
            "constant_gate_test_epoch18.pkl",
            "pose_gated_top4=outputs/maptr/visualization_predictions/"
            "pose_gated_top4_test_epoch18.pkl",
        ]
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--pred-pkl must be MODEL=PATH, got: {spec}")
        name, path = spec.split("=", 1)
        parsed.append((name.strip(), Path(path.strip())))
    return parsed


def load_gt_records(gt_json_path):
    gt_obj = load_json(gt_json_path)
    if not isinstance(gt_obj, dict) or "GTs" not in gt_obj:
        raise ValueError(f"Unsupported GT json structure: {gt_json_path}")
    records = gt_obj["GTs"]
    token_to_gt = {}
    for index, item in enumerate(records):
        token = item.get("sample_token") or item.get("token") or str(index)
        token_to_gt[token] = item
    return gt_obj, records, token_to_gt


def load_info_tokens(infos_pkl_path):
    obj = load_pickle(infos_pkl_path)
    infos = obj.get("infos", obj) if isinstance(obj, dict) else obj
    original_infos = list(infos)
    sorted_infos = sorted(original_infos, key=lambda item: item.get("timestamp", 0))
    original_tokens = [info.get("token") for info in original_infos]
    sorted_tokens = [info.get("token") for info in sorted_infos]
    return obj, original_infos, original_tokens, sorted_tokens


def class_name_for_vector(vector):
    name = vector.get("cls_name") or vector.get("class_name")
    if name:
        return name
    label = vector.get("label", vector.get("type"))
    if isinstance(label, str):
        return label
    if isinstance(label, (int, np.integer)) and 0 <= int(label) < len(CLASS_NAMES):
        return CLASS_NAMES[int(label)]
    return "unknown"


def normalize_vector(vector):
    pts = vector.get("pts", vector.get("points"))
    if pts is None:
        return None
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return None
    score = vector.get("confidence_level", vector.get("score", 1.0))
    return {
        "pts": pts[:, :2],
        "cls_name": class_name_for_vector(vector),
        "score": float(score),
    }


def class_formatted_to_records(obj, gt_records):
    pred_by_class = obj[0]
    records = []
    for index, gt_item in enumerate(gt_records):
        token = gt_item.get("sample_token") or gt_item.get("token") or str(index)
        vectors = []
        for cls_name in CLASS_NAMES:
            if cls_name not in pred_by_class:
                continue
            arr = np.asarray(pred_by_class[cls_name][index])
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            for row in arr:
                if len(row) < 5:
                    continue
                if len(row) % 2 == 1:
                    pts_flat = row[:-1]
                    score = float(row[-1])
                else:
                    pts_flat = row
                    score = 1.0
                pts = np.asarray(pts_flat, dtype=float).reshape(-1, 2)
                vectors.append(
                    {
                        "pts": pts.tolist(),
                        "pts_num": len(pts),
                        "cls_name": cls_name,
                        "confidence_level": score,
                    }
                )
        records.append({"sample_token": token, "vectors": vectors})
    return {"meta": {"source": "class_formatted.pkl"}, "results": records}


def load_prediction_records(path, gt_records):
    obj = load_any(path)
    structure = describe_structure(obj)
    if isinstance(obj, dict) and "results" in obj:
        records = obj["results"]
    elif (
        isinstance(obj, list)
        and len(obj) >= 1
        and isinstance(obj[0], dict)
        and all(cls_name in obj[0] for cls_name in CLASS_NAMES)
    ):
        converted = class_formatted_to_records(obj, gt_records)
        records = converted["results"]
        structure += "; interpreted as class-formatted [preds, gts]"
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        records = obj
    else:
        raise ValueError(f"Unsupported prediction structure in {path}: {structure}")

    token_to_pred = {}
    for index, item in enumerate(records):
        token = item.get("sample_token") or item.get("token")
        if token is None and index < len(gt_records):
            token = gt_records[index].get("sample_token") or str(index)
        token_to_pred[str(token)] = item
    return obj, records, token_to_pred, structure


def describe_structure(obj):
    if isinstance(obj, dict):
        parts = [f"dict keys={list(obj.keys())[:8]}"]
        if isinstance(obj.get("results"), list):
            parts.append(f"results_len={len(obj['results'])}")
            if obj["results"]:
                first = obj["results"][0]
                parts.append(f"first_result_keys={list(first.keys())[:8]}")
                parts.append(f"first_vectors={len(first.get('vectors', []))}")
        return "; ".join(parts)
    if isinstance(obj, list):
        parts = [f"list len={len(obj)}"]
        if obj:
            parts.append(f"first_type={type(obj[0]).__name__}")
            if isinstance(obj[0], dict):
                parts.append(f"first_keys={list(obj[0].keys())[:8]}")
        return "; ".join(parts)
    return type(obj).__name__


def parse_pc_range_from_config(config_path):
    seen = set()

    def visit(path):
        path = Path(path)
        if not path.exists() or path in seen:
            return None
        seen.add(path)
        text = path.read_text()
        match = re.search(r"pc_range\s*=\s*\[([^\]]+)\]", text)
        if match:
            return ast.literal_eval("[" + match.group(1) + "]")

        base_match = re.search(r"^_base_\s*=\s*(.+)$", text, re.MULTILINE)
        if not base_match:
            return None
        try:
            base_value = ast.literal_eval(base_match.group(1))
        except (SyntaxError, ValueError):
            return None
        base_paths = base_value if isinstance(base_value, list) else [base_value]
        for base in base_paths:
            found = visit((path.parent / base).resolve())
            if found:
                return found
        return None

    return visit(config_path)


def map_range_from_infos(infos):
    if not infos:
        return None
    item = infos[0].get("maptr_map_range")
    if not isinstance(item, dict):
        return None
    return [
        -float(item.get("left", 15.0)),
        -float(item.get("rear", 30.0)),
        float(item.get("right", 15.0)),
        float(item.get("front", 30.0)),
    ]


def resolve_map_range(args, infos):
    if args.map_range:
        return list(args.map_range), "argument --map-range"
    pc_range = parse_pc_range_from_config(args.config)
    if pc_range and len(pc_range) >= 6:
        return [pc_range[0], pc_range[1], pc_range[3], pc_range[4]], (
            f"pc_range from {args.config}: {pc_range}"
        )
    info_range = map_range_from_infos(infos)
    if info_range:
        return info_range, "maptr_map_range from infos pkl"
    return [-15.0, -30.0, 15.0, 30.0], "fallback default"


def selected_indices(args, total):
    if args.sample_indices:
        indices = args.sample_indices
    else:
        indices = list(range(min(args.num_samples, total)))
    bad = [idx for idx in indices if idx < 0 or idx >= total]
    if bad:
        raise IndexError(f"Sample index out of range: {bad}")
    return indices


def vectors_for_item(item, score_thr=None):
    vectors = []
    for vector in item.get("vectors", []):
        normalized = normalize_vector(vector)
        if normalized is None:
            continue
        if score_thr is not None and normalized["score"] < score_thr:
            continue
        vectors.append(normalized)
    return vectors


def draw_panel(
    ax,
    title,
    vectors,
    axis_range,
    is_gt=False,
    overlay_gt_vectors=None,
    style=None,
):
    style = style or DEFAULT_STYLE
    ax.set_title(title, fontsize=style["panel_title_fontsize"])
    ax.set_xlim(axis_range[0], axis_range[2])
    ax.set_ylim(axis_range[1], axis_range[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x right (m)", fontsize=style["axis_fontsize"])
    ax.set_ylabel("y forward (m)", fontsize=style["axis_fontsize"])
    ax.tick_params(axis="both", labelsize=style["tick_fontsize"])
    ax.grid(
        True,
        color="#dddddd",
        linewidth=style["grid_linewidth"],
        alpha=style["grid_alpha"],
    )
    ax.scatter([0], [0], marker="^", s=28, color="#111111", zorder=5)
    if overlay_gt_vectors:
        for vector in overlay_gt_vectors:
            pts = vector["pts"]
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color="#b5b5b5",
                linestyle="-",
                linewidth=style["gt_overlay_linewidth"],
                alpha=style["gt_overlay_alpha"],
                zorder=1,
            )
    for cls_name in CLASS_NAMES:
        cls_vectors = [vec for vec in vectors if vec["cls_name"] == cls_name]
        if is_gt:
            color, linestyle = GT_STYLES[cls_name]
            linewidth = style["gt_panel_linewidth"]
            alpha = style["gt_panel_alpha"]
        else:
            color = PRED_COLORS[cls_name]
            linestyle = PRED_STYLES[cls_name]
            linewidth = style["pred_linewidth"]
            alpha = style["pred_alpha"]
        for vector in cls_vectors:
            pts = vector["pts"]
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
            )


def legend_handles(style=None):
    style = style or DEFAULT_STYLE
    handles = [
        Line2D(
            [0],
            [0],
            color="#111111",
            linestyle="-",
            linewidth=style["gt_panel_linewidth"],
            label="GT divider",
        ),
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle="--",
            linewidth=style["gt_panel_linewidth"],
            label="GT boundary",
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle=":",
            linewidth=style["gt_panel_linewidth"] + 0.4,
            label="GT ped_crossing",
        ),
    ]
    for cls_name in CLASS_NAMES:
        handles.append(
            Line2D(
                [0],
                [0],
                color=PRED_COLORS[cls_name],
                linestyle=PRED_STYLES[cls_name],
                linewidth=style["pred_linewidth"],
                label=f"Pred {cls_name}",
            )
        )
    return handles


def make_figure(
    index,
    token,
    gt_item,
    pred_items,
    model_order,
    axis_range,
    score_thr,
    overlay_gt=False,
    style=None,
    title=None,
    hide_token=False,
):
    style = style or DEFAULT_STYLE
    gt_vectors = vectors_for_item(gt_item, score_thr=None)
    panels = [("GT", gt_vectors, True)]
    for model_name in model_order:
        item = pred_items[model_name].get(token, {"vectors": []})
        panel_title = DISPLAY_NAMES.get(model_name, model_name)
        panels.append((panel_title, vectors_for_item(item, score_thr=score_thr), False))

    figure_title = title
    if figure_title is None:
        figure_title = f"Sample {index:06d}"
        if not hide_token:
            figure_title += f" | {token}"

    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(style["fig_width"], style["fig_height"]),
        squeeze=False,
    )
    for ax, (panel_title, vectors, is_gt) in zip(axes[0], panels):
        draw_panel(
            ax,
            panel_title,
            vectors,
            axis_range,
            is_gt=is_gt,
            overlay_gt_vectors=gt_vectors if overlay_gt and not is_gt else None,
            style=style,
        )
    fig.suptitle(figure_title, fontsize=style["title_fontsize"])
    fig.legend(
        handles=legend_handles(style),
        loc="lower center",
        ncol=6,
        fontsize=style["legend_fontsize"],
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    if style["wspace"] is not None:
        fig.subplots_adjust(
            left=style["subplot_left"],
            right=style["subplot_right"],
            top=style["subplot_top"],
            bottom=style["subplot_bottom"],
            wspace=style["wspace"],
        )
    else:
        fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    return fig


def write_report(
    path,
    gt_json_path,
    infos_pkl_path,
    pred_specs,
    pred_structures,
    gt_structure,
    infos_structure,
    matching_lines,
    axis_range,
    map_range_source,
    output_paths,
    score_thr,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SimV2I-HD MapTR BEV Visualization Report",
        "",
        "## Inputs",
        "",
        f"- GT JSON: {gt_json_path}",
        f"- infos PKL: {infos_pkl_path}",
        "",
        "## Checkpoints",
        "",
    ]
    for model_name, _ in pred_specs:
        lines.append(f"- {model_name}: {CHECKPOINTS.get(model_name, 'unknown')}")
    lines.extend(
        [
            "",
            "## Prediction PKL Generation",
            "",
            "- The run script creates these pkl files from existing "
            "nuscmap_results.json files when they are missing.",
        ]
    )
    for model_name, pred_path in pred_specs:
        lines.append(f"- {model_name}: {pred_path}")
    lines.extend(
        [
            "",
            "## Structures",
            "",
            f"- GT: {gt_structure}",
            f"- infos: {infos_structure}",
        ]
    )
    for model_name, structure in pred_structures.items():
        lines.append(f"- {model_name}: {structure}")
    lines.extend(
        [
            "",
            "## Matching",
            "",
            "- Sample selection uses GT list indices; prediction panels are "
            "matched by sample_token.",
        ]
    )
    lines.extend(f"- {line}" for line in matching_lines)
    lines.extend(
        [
            "",
            "## Coordinate System And Map Range",
            "",
            "- Coordinate system: ego-centered BEV, x right, y forward.",
            f"- Map range: x=[{axis_range[0]}, {axis_range[2]}], "
            f"y=[{axis_range[1]}, {axis_range[3]}].",
            f"- Map range source: {map_range_source}.",
            "",
            "## Outputs",
            "",
            f"- Score threshold: {score_thr}",
            f"- Generated PNG count: {len(output_paths)}",
            f"- Example output path: {output_paths[0] if output_paths else 'N/A'}",
            "",
            "## Known Limitations",
            "",
            "- This is qualitative visualization; it does not compute AP or Chamfer metrics.",
            "- Prediction clutter depends on the chosen score threshold.",
            "- Existing checkpoints, data, and eval outputs are read only by this workflow.",
            "- If a class-formatted pkl is used, token matching falls back to GT order.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    gt_obj, gt_records, token_to_gt = load_gt_records(args.gt_json)
    infos_obj, infos, info_tokens, sorted_info_tokens = load_info_tokens(
        args.infos_pkl
    )
    pred_specs = parse_pred_specs(args.pred_pkl)

    pred_items = {}
    pred_structures = {}
    matching_lines = []
    gt_tokens = [item.get("sample_token") or item.get("token") for item in gt_records]
    matching_lines.append(f"GT tokens vs infos pkl order: {gt_tokens == info_tokens}")
    matching_lines.append(
        f"GT tokens vs timestamp-sorted infos order: {gt_tokens == sorted_info_tokens}"
    )

    for model_name, pred_path in pred_specs:
        _, records, token_to_pred, structure = load_prediction_records(
            pred_path, gt_records
        )
        pred_items[model_name] = token_to_pred
        pred_structures[model_name] = structure
        matched = sum(1 for token in gt_tokens if token in token_to_pred)
        pred_tokens = [item.get("sample_token") or item.get("token") for item in records]
        matching_lines.append(
            f"{model_name}: matched {matched}/{len(gt_tokens)} samples by sample_token; "
            f"prediction order equals GT order: {pred_tokens == gt_tokens}"
        )

    axis_range, map_range_source = resolve_map_range(args, infos)
    indices = selected_indices(args, len(gt_records))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_order = [model_name for model_name, _ in pred_specs]
    if args.paper_4panel:
        model_order = [
            model_name
            for model_name in ("ego_only", "dynamic_top4", "pose_gated_top4")
            if model_name in pred_items
        ]
    elif args.paper_5panel:
        model_order = [
            model_name
            for model_name in (
                "ego_only",
                "dynamic_top4",
                "constant_gate",
                "pose_gated_top4",
            )
            if model_name in pred_items
        ]
    short_name = "_".join(
        MODEL_SHORT_NAMES.get(model_name, model_name) for model_name in model_order
    )
    panel_count = 1 + len(model_order)
    style = resolve_style(args, panel_count)
    apply_text_style(style)

    output_paths = []
    for index in indices:
        gt_item = gt_records[index]
        token = gt_item.get("sample_token") or gt_item.get("token") or str(index)
        figure_title = args.short_title
        if figure_title is None and args.paper_ready:
            figure_title = f"Sample {index}"
        fig = make_figure(
            index,
            token,
            gt_item,
            pred_items,
            model_order,
            axis_range,
            args.score_thr,
            overlay_gt=args.overlay_gt,
            style=style,
            title=figure_title,
            hide_token=args.hide_token or args.paper_ready,
        )
        suffix = "_overlay" if args.overlay_gt else ""
        if args.output_name and len(indices) == 1:
            out_base = output_dir / args.output_name
        elif args.output_name:
            out_base = output_dir / f"{args.output_name}_sample_{index:06d}"
        else:
            out_base = (
                output_dir
                / f"{args.filename_prefix}sample_{index:06d}_gt_{short_name}{suffix}"
            )
        out_path = out_base.with_suffix(".png")
        fig.savefig(out_path, dpi=200)
        if args.save_pdf:
            fig.savefig(out_base.with_suffix(".pdf"))
        plt.close(fig)
        output_paths.append(str(out_path))

    write_report(
        args.report_path,
        args.gt_json,
        args.infos_pkl,
        pred_specs,
        pred_structures,
        describe_structure(gt_obj),
        describe_structure(infos_obj),
        matching_lines,
        axis_range,
        map_range_source,
        output_paths,
        args.score_thr,
    )

    print(f"Generated {len(output_paths)} PNG files in {output_dir}")
    if output_paths:
        print(f"Example: {output_paths[0]}")
    print(f"Report: {args.report_path}")


if __name__ == "__main__":
    main()
