#!/usr/bin/env python
"""Collect simulator-derived visual candidates for the method overview figure."""

import argparse
import json
import math
import os
import pickle
import shutil
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-maptr")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/maptr-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SIMV2I_DATA_ROOT = Path(
    os.environ.get(
        "SIMV2I_HD_ROOT",
        ROOT / "data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k",
    )
)
INFO_PKL = SIMV2I_DATA_ROOT / "simv2i_maptr_infos_test.pkl"
RAW_ROOT = Path(os.environ.get("SIMV2I_HD_RAW_ROOT", ROOT / "data/raw"))
RSU_LAYOUT_METADATA = (
    RAW_ROOT
    / "simv2i_hd_benchmark_v2_dynamic_rsu_town10_r040_clear_heavy_"
    "4rsu_dynamic_top4_vqprior_strict/metadata/run.json"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "outputs/maptr/paper_assets/method_overview_candidates"
)
DEFAULT_REPORT_PATH = (
    ROOT / "outputs/maptr/paper_assets/METHOD_OVERVIEW_ASSET_SEARCH_REPORT.md"
)

EGO_VIEW_ORDER = [
    ("CAM_FRONT_LEFT", "front-left"),
    ("CAM_FRONT", "front"),
    ("CAM_FRONT_RIGHT", "front-right"),
    ("CAM_BACK_LEFT", "back-left"),
    ("CAM_BACK", "back"),
    ("CAM_BACK_RIGHT", "back-right"),
]
PANEL_DIRS = {
    "panel1": "panel1_input_scene",
    "panel2": "panel2_dynamic_rsu_selection",
    "panel3": "panel3_pose_gated_fusion",
    "panel4": "panel4_hd_map_output",
}
SAMPLE_ALIASES = {
    3608: "sample3608",
    3865: "sample3865",
    3783: "sample3783",
    3624: "sample3624",
    938: "sample0938",
    0: "sample0000",
}
INPUT_SCENE_SAMPLES = [3608, 3865]
FUSION_SAMPLES = [3608, 3865]
HD_MAP_SAMPLES = [3624, 3608, 3865, 3783]
CLASS_NAMES = ("divider", "boundary", "ped_crossing")
CLASS_COLORS = {
    "divider": "#1f77b4",
    "boundary": "#d55e00",
    "ped_crossing": "#009e73",
}
CLASS_LABELS = {0: "divider", 1: "boundary", 2: "ped_crossing"}


def rel(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT)
    except ValueError:
        return path.resolve()


def load_infos(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return obj.get("infos", obj.get("data_list", obj))
    return obj


def load_font(size=18):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def ensure_dirs(output_root):
    dirs = {}
    for key, name in PANEL_DIRS.items():
        dirs[key] = output_root / name
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs


def copy_asset(src, dst, manifest, panel, note):
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append(
        {
            "panel": panel,
            "asset_path": str(rel(dst)),
            "source_path": str(rel(src)),
            "note": note,
        }
    )
    return dst


def make_contact_sheet(items, out_path, columns=3, tile=(360, 210), title=None):
    items = [(Path(path), label) for path, label in items if Path(path).exists()]
    if not items:
        return None
    label_h = 34
    title_h = 42 if title else 0
    pad = 14
    rows = math.ceil(len(items) / columns)
    width = columns * tile[0] + (columns + 1) * pad
    height = title_h + rows * (tile[1] + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(22)
    label_font = load_font(15)
    if title:
        draw.text((pad, 10), title, fill=(25, 25, 25), font=title_font)
    y0 = title_h + pad
    for idx, (path, label) in enumerate(items):
        col = idx % columns
        row = idx // columns
        x = pad + col * (tile[0] + pad)
        y = y0 + row * (tile[1] + label_h + pad)
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail(tile, Image.Resampling.LANCZOS)
            bg = Image.new("RGB", tile, (248, 248, 248))
            ox = (tile[0] - im.width) // 2
            oy = (tile[1] - im.height) // 2
            bg.paste(im, (ox, oy))
        canvas.paste(bg, (x, y))
        draw.rectangle((x, y, x + tile[0], y + tile[1]), outline=(210, 210, 210))
        draw.text((x + 4, y + tile[1] + 7), label[:55], fill=(35, 35, 35), font=label_font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def sample_alias(index):
    return SAMPLE_ALIASES.get(index, f"sample{index:04d}")


def image_path(info, cam_name):
    return ROOT / info["cams"][cam_name]["data_path"]


def rsu_image_items(info):
    return [
        (ROOT / cam_info["data_path"], rsu_id.replace("_rgb", ""))
        for rsu_id, cam_info in info.get("rsu_cams", {}).items()
    ]


def make_ego_contact(info, sample_idx, out_dir, manifest, panel="panel1"):
    alias = sample_alias(sample_idx)
    items = []
    for cam_name, label in EGO_VIEW_ORDER:
        path = image_path(info, cam_name)
        items.append((path, f"{alias} {label}"))
    out = out_dir / f"{alias}_ego_6view_contact.png"
    make_contact_sheet(items, out, columns=3, tile=(360, 205), title=f"{alias} ego 6-view")
    manifest.append(
        {
            "panel": panel,
            "asset_path": str(rel(out)),
            "source_path": "six ego RGB images from infos_test.pkl",
            "note": "CARLA ego-camera contact sheet.",
        }
    )
    return out


def make_rsu_contact(info, sample_idx, out_dir, manifest, panel, suffix="selected_rsu_top4"):
    alias = sample_alias(sample_idx)
    out = out_dir / f"{alias}_{suffix}.png"
    make_contact_sheet(
        rsu_image_items(info),
        out,
        columns=4,
        tile=(300, 170),
        title=f"{alias} selected top-4 RSU views",
    )
    manifest.append(
        {
            "panel": panel,
            "asset_path": str(rel(out)),
            "source_path": "selected top-4 RSU RGB images from infos_test.pkl",
            "note": "Frame-matched infrastructure camera thumbnails.",
        }
    )
    return out


def make_fusion_contact(info, sample_idx, out_dir, manifest):
    alias = sample_alias(sample_idx)
    items = []
    for cam_name, label in EGO_VIEW_ORDER:
        items.append((image_path(info, cam_name), f"ego {label}"))
    for path, label in rsu_image_items(info):
        items.append((path, f"top-4 {label}"))
    out = out_dir / f"{alias}_ego6_rsu4_fusion_contact.png"
    make_contact_sheet(
        items,
        out,
        columns=5,
        tile=(290, 165),
        title=f"{alias} ego 6-view + selected top-4 RSU",
    )
    manifest.append(
        {
            "panel": "panel3",
            "asset_path": str(rel(out)),
            "source_path": "ego + selected RSU RGB images from infos_test.pkl",
            "note": "Pose-gated V2I fusion input thumbnail candidate.",
        }
    )
    return out


def selected_rsu_ids(info):
    return list(info.get("rsu_cams", {}).keys())


def load_rsu_layout():
    with open(RSU_LAYOUT_METADATA) as f:
        metadata = json.load(f)
    config = metadata["config"]
    transforms = config["sensors"]["infrastructure"]["transforms"]
    dynamic = config["dynamic_rsu"]
    return transforms, dynamic


def quat_yaw_rad(q):
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def draw_rsu_points(ax, transforms, selected_ids, blocked_ids, show_selected):
    selected_set = set(selected_ids)
    for transform in transforms:
        is_selected = transform["id"] in selected_set
        is_blocked = transform["id"] in blocked_ids
        color = "#c9c9c9" if not is_blocked else "#eeeeee"
        edge = "#777777" if not is_blocked else "#b5b5b5"
        size = 34 if not is_selected else 120
        z = 3 if not is_selected else 6
        if show_selected and is_selected:
            color = "#ff8c1a"
            edge = "#5c2a00"
        ax.scatter(
            transform["x"],
            transform["y"],
            s=size,
            c=color,
            edgecolors=edge,
            linewidths=0.8,
            zorder=z,
        )
        yaw = math.radians(transform["yaw"])
        length = 6.0 if not is_selected else 10.0
        alpha = 0.35 if not is_selected else 0.9
        ax.arrow(
            transform["x"],
            transform["y"],
            math.cos(yaw) * length,
            math.sin(yaw) * length,
            width=0.35 if not is_selected else 0.65,
            head_width=2.4 if not is_selected else 3.8,
            head_length=2.8 if not is_selected else 4.2,
            color="#777777" if not is_selected else "#5c2a00",
            alpha=alpha,
            length_includes_head=True,
            zorder=z,
        )
        if show_selected and is_selected:
            rank = selected_ids.index(transform["id"]) + 1
            ax.text(
                transform["x"] + 3,
                transform["y"] + 3,
                f"{rank}:{transform['id'].replace('_rgb', '')}",
                fontsize=8,
                weight="bold",
                color="#4a2300",
                zorder=7,
            )


def draw_ego(ax, info):
    x, y = info["ego2global_translation"][:2]
    yaw = quat_yaw_rad(info["ego2global_rotation"])
    ax.scatter(x, y, s=160, marker="*", c="#1565c0", edgecolors="white", linewidths=1.1, zorder=8)
    ax.arrow(
        x,
        y,
        math.cos(yaw) * 12,
        math.sin(yaw) * 12,
        width=0.8,
        head_width=4.5,
        head_length=5.0,
        color="#1565c0",
        length_includes_head=True,
        zorder=8,
    )
    ax.text(x + 3, y - 6, "ego", fontsize=9, weight="bold", color="#0b3d73", zorder=9)


def setup_layout_axis(ax, transforms, info, title):
    xs = [t["x"] for t in transforms] + [info["ego2global_translation"][0]]
    ys = [t["y"] for t in transforms] + [info["ego2global_translation"][1]]
    margin = 22
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#dddddd", linewidth=0.45)
    ax.set_xlabel("CARLA world x (m)", fontsize=9)
    ax.set_ylabel("CARLA world y (m)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=12, weight="bold")


def make_rsu_layout_figure(info, sample_idx, out_dir, manifest):
    alias = sample_alias(sample_idx)
    transforms, dynamic = load_rsu_layout()
    selected_ids = selected_rsu_ids(info)
    blocked_ids = set(dynamic["visual_quality_prior"]["blocked_candidate_ids"])

    def single_plot(path, show_selected, title):
        fig, ax = plt.subplots(figsize=(6.0, 6.2), dpi=220)
        setup_layout_axis(ax, transforms, info, title)
        draw_rsu_points(ax, transforms, selected_ids, blocked_ids, show_selected)
        draw_ego(ax, info)
        if show_selected:
            ex, ey = info["ego2global_translation"][:2]
            by_id = {t["id"]: t for t in transforms}
            for rsu_id in selected_ids:
                if rsu_id in by_id:
                    t = by_id[rsu_id]
                    ax.plot([ex, t["x"]], [ey, t["y"]], color="#ff8c1a", linewidth=1.0, alpha=0.65)
        handles = [
            plt.Line2D([], [], marker="o", linestyle="", color="#c9c9c9", markeredgecolor="#777777", label="32 candidates"),
            plt.Line2D([], [], marker="*", linestyle="", color="#1565c0", markeredgecolor="white", markersize=11, label="ego"),
        ]
        if show_selected:
            handles.append(
                plt.Line2D([], [], marker="o", linestyle="", color="#ff8c1a", markeredgecolor="#5c2a00", label="selected top-4")
            )
        ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    all_path = out_dir / f"{alias}_rsu_layout_all_candidates.png"
    selected_path = out_dir / f"{alias}_rsu_layout_top4_selected.png"
    single_plot(all_path, False, f"{alias}: 32 candidate RSUs")
    single_plot(selected_path, True, f"{alias}: dynamic top-4 selected")

    before_after = out_dir / f"{alias}_rsu_layout_before_after.png"
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.3), dpi=220)
    for ax, show, title in [
        (axes[0], False, "Before selection"),
        (axes[1], True, "After dynamic top-4"),
    ]:
        setup_layout_axis(ax, transforms, info, title)
        draw_rsu_points(ax, transforms, selected_ids, blocked_ids, show)
        draw_ego(ax, info)
        if show:
            ex, ey = info["ego2global_translation"][:2]
            by_id = {t["id"]: t for t in transforms}
            for rsu_id in selected_ids:
                if rsu_id in by_id:
                    t = by_id[rsu_id]
                    ax.plot([ex, t["x"]], [ey, t["y"]], color="#ff8c1a", linewidth=1.0, alpha=0.65)
    fig.suptitle(f"{alias}: CARLA RSU candidate pool and selected top-4", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(before_after)
    plt.close(fig)

    for path, note in [
        (all_path, "All 32 CARLA RSU candidate positions and viewing directions."),
        (selected_path, "Same layout with the sample-level selected top-4 highlighted."),
        (before_after, "Before/after dynamic RSU selection contrast."),
    ]:
        manifest.append(
            {
                "panel": "panel2",
                "asset_path": str(rel(path)),
                "source_path": str(rel(RSU_LAYOUT_METADATA)),
                "note": note,
            }
        )
    return [all_path, selected_path, before_after]


def make_rsu_frequency(infos, out_dir, manifest):
    counter = Counter()
    for info in infos:
        counter.update(selected_rsu_ids(info))
    rows = sorted(counter.items(), key=lambda item: int(item[0].split("_")[1]))
    csv_path = out_dir / "test_selected_rsu_frequency.csv"
    with open(csv_path, "w") as f:
        f.write("rsu_id,selected_count\n")
        for rsu_id, count in rows:
            f.write(f"{rsu_id},{count}\n")
    ids = [k.replace("_rgb", "") for k, _ in rows]
    counts = [v for _, v in rows]
    fig, ax = plt.subplots(figsize=(9.5, 3.8), dpi=220)
    ax.bar(ids, counts, color="#607d8b")
    ax.set_ylabel("selected frames", fontsize=9)
    ax.set_title("Dynamic top-4 RSU selection frequency in test split", fontsize=11, weight="bold")
    ax.tick_params(axis="x", labelrotation=70, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", color="#dddddd", linewidth=0.4)
    fig.tight_layout()
    png_path = out_dir / "test_selected_rsu_frequency.png"
    fig.savefig(png_path)
    plt.close(fig)
    for path, note in [
        (png_path, "Test split frequency of selected top-4 RSU IDs."),
        (csv_path, "CSV backing the selected-RSU frequency asset."),
    ]:
        manifest.append(
            {
                "panel": "panel2",
                "asset_path": str(rel(path)),
                "source_path": str(rel(INFO_PKL)),
                "note": note,
            }
        )
    return png_path


def draw_compact_gt_map(info, sample_idx, out_dir, manifest):
    alias = sample_alias(sample_idx)
    labels = info["maptr_gt_labels"]
    points = info["maptr_gt_fixed_points"]
    counts = info.get("maptr_gt_category_counts", {})
    fig, ax = plt.subplots(figsize=(3.1, 4.35), dpi=260)
    for label, pts in zip(labels, points):
        cls = CLASS_LABELS.get(int(label), str(label))
        arr = np.asarray(pts, dtype=float)
        ax.plot(
            arr[:, 0],
            arr[:, 1],
            color=CLASS_COLORS.get(cls, "#555555"),
            linewidth=1.5 if cls != "ped_crossing" else 1.8,
            alpha=0.96,
            solid_capstyle="round",
        )
    ax.scatter([0], [0], marker="^", s=55, color="#222222", zorder=5)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-30, 30)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e0e0e0", linewidth=0.35)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        f"{alias} GT vectors | div {counts.get('divider', 0)}, "
        f"bound {counts.get('boundary', 0)}, ped {counts.get('ped_crossing', 0)}",
        fontsize=8.5,
    )
    fig.tight_layout()
    out = out_dir / f"{alias}_gt_vector_map_compact.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    manifest.append(
        {
            "panel": "panel4",
            "asset_path": str(rel(out)),
            "source_path": str(rel(ROOT / info["maptr_gt_path"])),
            "note": "Compact GT vector-map rendering for method overview output panel.",
        }
    )
    return out


def write_report(report_path, output_root, manifest, contact_sheets):
    panel1 = output_root / PANEL_DIRS["panel1"]
    panel2 = output_root / PANEL_DIRS["panel2"]
    panel3 = output_root / PANEL_DIRS["panel3"]
    panel4 = output_root / PANEL_DIRS["panel4"]
    lines = [
        "# Method Overview Asset Search Report",
        "",
        "## Scope",
        "",
        "This search collected simulator/data-derived visual assets for a Method overview figure. No training results, checkpoints, prediction PKLs, raw data, or LaTeX files were modified. All assets below are copies or newly rendered index/summary images under `outputs/maptr/paper_assets/`.",
        "",
        "## Investigated Paths",
        "",
        "- `${MAPTR_ROOT}/data/raw`",
        "- `${MAPTR_ROOT}/data/maptr`",
        "- `${MAPTR_ROOT}/outputs`",
        "- `${MAPTR_ROOT}/outputs/maptr/visualizations`",
        "- `${MAPTR_ROOT}/outputs/maptr/visualizations/qualitative_ranking`",
        "- `${MAPTR_ROOT}/outputs/maptr/visualizations/bev_map_comparison`",
        "- `${MAPTR_ROOT}/tools`",
        "- `${MAPTR_ROOT}/scripts`",
        "- `${ACCV_TEMPLATE_ROOT}` (optional read-only check; only existing qualitative PDF was found in `figures/`)",
        "",
        "## Key Findings",
        "",
        "- `simv2i_maptr_infos_test.pkl` contains frame-matched ego 6-view camera paths, selected top-4 RSU image paths, RSU extrinsics, and GT vector-map annotations.",
        "- Dynamic v2 raw metadata contains the 32-RSU CARLA candidate pool in `metadata/run.json` and records the policy as `geometry_roi_coverage_top4_vqprior`.",
        "- For the 20k test samples used in the paper figure, the raw run stores selected top-4 RSU images only; the 32-candidate layout is recovered from the shared dynamic-v2 run metadata.",
        "- No separate third-person/spectator ego-vehicle render was found in the searched project paths; input-scene candidates therefore use actual ego/RSU RGB simulator camera frames.",
        "",
        "## Panel Recommendations",
        "",
        "### Panel 1: Input Scene",
        "",
        f"- **1st**: `{rel(panel1 / 'sample3608_ego_6view_contact.png')}`",
        "  - Good because it is the exact CARLA ego 6-view sample corresponding to the qualitative figure.",
        "  - Strength: directly shows simulator RGB data rather than generic concept art.",
        "  - Limitation: not a third-person view, so the ego vehicle itself is implicit from camera pose.",
        "  - Crop/annotation: crop to front/front-left/front-right or keep as miniature 6-view strip.",
        f"- **2nd**: `{rel(panel1 / 'sample3865_ego_6view_contact.png')}`",
        "  - Good as an alternate sunset dense sample with matched RSU/map data.",
        f"- **Backup**: `{rel(panel1 / 'sample3608_ego_front_raw.png')}`",
        "  - Use when the overview panel needs a single simple road scene.",
        "",
        "### Panel 2: Dynamic RSU Selection",
        "",
        f"- **1st**: `{rel(panel2 / 'sample3608_rsu_layout_before_after.png')}`",
        "  - Good because it shows all 32 CARLA candidate RSUs distributed across Town10HD-style world coordinates, then highlights the sample-level selected top-4.",
        "  - Strength: avoids the misleading impression that all 32 cameras stare at one point; each RSU has its own position and yaw arrow.",
        "  - Limitation: it is a metadata-derived top-down plot, not a CARLA RGB render.",
        "  - Crop/annotation: crop the two subplots tightly, or use only the right-side after-selection panel if space is tight.",
        f"- **2nd**: `{rel(panel2 / 'sample3608_rsu_layout_top4_selected.png')}`",
        "  - Good single-panel version for compact method diagrams.",
        f"- **Backup**: `{rel(panel2 / 'sample3608_selected_rsu_top4.png')}` and `{rel(panel2 / 'test_selected_rsu_frequency.png')}`",
        "  - Use thumbnails to show selected views; use frequency only as an auxiliary dataset-statistic visual.",
        "",
        "### Panel 3: Pose-gated V2I Fusion",
        "",
        f"- **1st**: `{rel(panel3 / 'sample3608_ego6_rsu4_fusion_contact.png')}`",
        "  - Good because it shows the exact fusion inputs: ego 6-view plus selected top-4 RSU views for sample 3608.",
        "  - Strength: frame matched and simulator-derived.",
        "  - Limitation: it does not directly visualize learned gate weights.",
        "  - Crop/annotation: can crop into two rows, ego views on top and RSU views below.",
        f"- **2nd**: `{rel(panel3 / 'sample3865_ego6_rsu4_fusion_contact.png')}`",
        "  - Good alternate with a different selected RSU set.",
        f"- **Backup**: `{rel(panel3 / 'sample3608_selected_rsu_top4.png')}`",
        "",
        "### Panel 4: Vectorized HD Map Output",
        "",
        f"- **1st**: `{rel(panel4 / 'sample3624_gt_vector_map_compact.png')}`",
        "  - Good because divider, boundary, and ped_crossing are all visually clear while the scene is still compact: divider 20, boundary 5, ped_crossing 2.",
        "  - Strength: clearer class separation than the smallest sample, and simpler than many full qualitative examples.",
        "  - Limitation: GT-only, not a model prediction.",
        "  - Crop/annotation: already compact; add a tiny class legend in the final diagram if needed.",
        f"- **2nd**: `{rel(panel4 / 'sample3608_gt_vector_map_compact.png')}`",
        "  - Good because it matches the paper qualitative sample.",
        f"- **Backup**: `{rel(panel4 / 'sample3783_gt_vector_map_compact.png')}`, `{rel(panel4 / 'sample_3608_paper_ready_4panel.png')}`, and `{rel(panel4 / 'ranked_sample_003608_gt_ego_dynamic_pose_overlay.png')}`",
        "  - sample3783 is the simplest all-class map but its boundary instance is subtle; use the paper qualitative copies only if the overview should echo the qualitative comparison.",
        "",
        "## Candidate Asset Notes",
        "",
        "| Asset | Panel | Why suitable | Strength | Limitation | Crop/annotation |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| `{rel(panel1 / 'sample3608_ego_6view_contact.png')}` | 1 | Exact CARLA ego 6-view for the paper qualitative sample. | Simulator-derived, frame matched, shows intersection/road context. | Ego vehicle is implicit except partial body in rear-side view. | Crop to 3-view strip or keep full 6-view. |",
        f"| `{rel(panel1 / 'sample3865_ego_6view_contact.png')}` | 1 | Alternate CARLA road scene with visible traffic context. | Strong simulator texture and urban road view. | Different map sample from the main qualitative figure. | Crop to front row if space is tight. |",
        f"| `{rel(panel1 / 'sample3608_ego_front_raw.png')}` | 1 | Single raw ego front view for a compact input-scene icon. | Clean intersection/road signal. | No explicit multi-view context. | Add small camera/frustum annotation if used alone. |",
        f"| `{rel(panel2 / 'sample3608_rsu_layout_before_after.png')}` | 2 | Shows 32 candidates before selection and sample top-4 after selection. | Best at preventing the all-cameras-look-at-one-point misconception. | Metadata plot, not RGB render. | Crop tightly or use one half as needed. |",
        f"| `{rel(panel2 / 'sample3608_rsu_layout_top4_selected.png')}` | 2 | Single layout with all candidates and selected RSUs emphasized. | Compact and directly tied to sample 3608 selected IDs. | Labels near ego can be dense at small scale. | Enlarge selected labels or remove axes in final artwork. |",
        f"| `{rel(panel2 / 'sample3608_rsu_layout_all_candidates.png')}` | 2 | Shows the distributed 32-RSU candidate pool alone. | Good for the selection-before state. | Does not show top-4 outcome by itself. | Pair with selected layout or add callout. |",
        f"| `{rel(panel2 / 'sample3608_selected_rsu_top4.png')}` | 2/3 | Shows the actual four selected RSU RGB views. | Direct simulator evidence of selected infrastructure cameras. | Does not show 32-candidate spatial layout. | Use as thumbnail row below the layout. |",
        f"| `{rel(panel2 / 'test_selected_rsu_frequency.png')}` | 2 | Shows dataset-level selected-RSU distribution. | Useful auxiliary proof that selection is dynamic over IDs. | Less intuitive as a method-overview visual. | Use only as appendix or small inset. |",
        f"| `{rel(panel3 / 'sample3608_ego6_rsu4_fusion_contact.png')}` | 3 | Shows all frame-matched fusion inputs for sample 3608. | Best visual for pose-gated V2I input composition. | Does not show gate weights. | Split into ego row and RSU row for final figure. |",
        f"| `{rel(panel3 / 'sample3865_ego6_rsu4_fusion_contact.png')}` | 3 | Alternate fusion input sample with a different top-4 set. | Demonstrates sample-dependent selected RSUs. | Not the main qualitative sample. | Use as backup if sample3608 is visually busy. |",
        f"| `{rel(panel4 / 'sample3624_gt_vector_map_compact.png')}` | 4 | Clear divider, boundary, and ped_crossing in a compact scene. | Best balance of readability and simplicity. | GT-only, not a prediction. | Add class legend externally if needed. |",
        f"| `{rel(panel4 / 'sample3608_gt_vector_map_compact.png')}` | 4 | Compact output for the main qualitative sample. | Directly aligns with the paper-ready sample 3608 figure. | Slightly busier than sample3624. | Good if all panels should use one sample. |",
        f"| `{rel(panel4 / 'sample3865_gt_vector_map_compact.png')}` | 4 | Simple curved-road HD map candidate. | Very clean lane geometry. | Ped/boundary are less prominent than sample3624. | Use if a curved-road motif is preferred. |",
        f"| `{rel(panel4 / 'sample3783_gt_vector_map_compact.png')}` | 4 | Smallest all-class map found among inspected samples. | Minimal, easy to shrink. | Boundary is subtle, so three-class story is weaker. | Add colored annotation if used. |",
        f"| `{rel(panel4 / 'sample_3608_paper_ready_4panel.png')}` | 4 | Existing qualitative comparison figure. | Publication-ready and already checked. | Too wide for a tiny method output icon. | Crop one panel or use as reference only. |",
        f"| `{rel(panel4 / 'ranked_sample_003608_gt_ego_dynamic_pose_overlay.png')}` | 4 | Existing ranking overlay for sample 3608. | Shows model comparison context. | Long token/title and multi-panel layout are not overview-friendly. | Crop to GT or pose-gated panel if reused. |",
        "",
        "## Minimum Set Ready For Figure Redesign",
        "",
        f"- Panel 1: `{rel(panel1 / 'sample3608_ego_6view_contact.png')}`",
        f"- Panel 2: `{rel(panel2 / 'sample3608_rsu_layout_before_after.png')}`",
        f"- Panel 3: `{rel(panel3 / 'sample3608_ego6_rsu4_fusion_contact.png')}`",
        f"- Panel 4: `{rel(panel4 / 'sample3624_gt_vector_map_compact.png')}`",
        "",
        "## Copied / Generated Assets",
        "",
    ]
    for item in manifest:
        lines.append(f"- `{item['asset_path']}`")
        lines.append(f"  - Panel: {item['panel']}")
        lines.append(f"  - Source: `{item['source_path']}`")
        lines.append(f"  - Note: {item['note']}")
    lines.extend(["", "## Contact Sheets", ""])
    for path in contact_sheets:
        lines.append(f"- `{rel(path)}`")
    lines.extend(
        [
            "",
            "## Structure Redesign Suggestion",
            "",
            "Use actual simulator RGB frames for the Input Scene and Fusion panels, the metadata-derived 32-RSU layout plus selected top-4 highlight for Dynamic RSU Selection, and the compact GT vector map for the HD map output. This combination keeps every visual tied to the SimV2I-HD/CARLA dataset while avoiding generic stock-road imagery.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    args = parser.parse_args()

    output_root = Path(args.output_root)
    report_path = Path(args.report_path)
    dirs = ensure_dirs(output_root)
    infos = load_infos(INFO_PKL)
    manifest = []

    panel1_items = []
    for sample_idx in INPUT_SCENE_SAMPLES:
        info = infos[sample_idx]
        alias = sample_alias(sample_idx)
        contact = make_ego_contact(info, sample_idx, dirs["panel1"], manifest, panel="panel1")
        panel1_items.append((contact, f"{alias} ego 6-view"))
        front_copy = copy_asset(
            image_path(info, "CAM_FRONT"),
            dirs["panel1"] / f"{alias}_ego_front_raw.png",
            manifest,
            "panel1",
            "Single CARLA ego front RGB frame.",
        )
        if front_copy:
            panel1_items.append((front_copy, f"{alias} front"))

    panel2_items = []
    info3608 = infos[3608]
    panel2_items.extend((p, p.stem.replace("sample3608_", "")) for p in make_rsu_layout_figure(info3608, 3608, dirs["panel2"], manifest))
    rsu_contact = make_rsu_contact(info3608, 3608, dirs["panel2"], manifest, "panel2")
    panel2_items.append((rsu_contact, "sample3608 top-4 RSU views"))
    freq = make_rsu_frequency(infos, dirs["panel2"], manifest)
    panel2_items.append((freq, "test selected-RSU frequency"))

    panel3_items = []
    for sample_idx in FUSION_SAMPLES:
        info = infos[sample_idx]
        alias = sample_alias(sample_idx)
        fusion = make_fusion_contact(info, sample_idx, dirs["panel3"], manifest)
        panel3_items.append((fusion, f"{alias} ego+RSU"))
        rsu = make_rsu_contact(info, sample_idx, dirs["panel3"], manifest, "panel3")
        panel3_items.append((rsu, f"{alias} top-4 RSU"))

    panel4_items = []
    for sample_idx in HD_MAP_SAMPLES:
        out = draw_compact_gt_map(infos[sample_idx], sample_idx, dirs["panel4"], manifest)
        counts = infos[sample_idx].get("maptr_gt_category_counts", {})
        panel4_items.append(
            (
                out,
                f"{sample_alias(sample_idx)} GT d{counts.get('divider', 0)} "
                f"b{counts.get('boundary', 0)} p{counts.get('ped_crossing', 0)}",
            )
        )

    existing_assets = [
        (
            ROOT / "outputs/maptr/visualizations/paper_figures/sample_3608_paper_ready_4panel.png",
            dirs["panel4"] / "sample_3608_paper_ready_4panel.png",
            "Paper-ready 4-panel qualitative figure.",
        ),
        (
            ROOT / "outputs/maptr/visualizations/paper_figures/sample_3608_paper_ready_4panel.pdf",
            dirs["panel4"] / "sample_3608_paper_ready_4panel.pdf",
            "PDF version of paper-ready qualitative figure.",
        ),
        (
            ROOT / "outputs/maptr/visualizations/qualitative_ranking/figures/"
            "ranked_sample_003608_gt_ego_dynamic_pose_overlay.png",
            dirs["panel4"] / "ranked_sample_003608_gt_ego_dynamic_pose_overlay.png",
            "Existing qualitative ranking overlay for sample 3608.",
        ),
        (
            ROOT / "outputs/maptr/visualizations/qualitative_ranking/figures/"
            "ranked_sample_003865_gt_ego_dynamic_pose_overlay.png",
            dirs["panel4"] / "ranked_sample_003865_gt_ego_dynamic_pose_overlay.png",
            "Existing qualitative ranking overlay for sample 3865.",
        ),
    ]
    for src, dst, note in existing_assets:
        copied = copy_asset(src, dst, manifest, "panel4", note)
        if copied and copied.suffix.lower() == ".png":
            panel4_items.append((copied, copied.stem))

    contact_sheets = []
    for name, items, cols, tile, title in [
        ("panel1_contact_sheet.png", panel1_items, 2, (420, 235), "Panel 1: Input Scene Candidates"),
        ("panel2_contact_sheet.png", panel2_items, 2, (470, 285), "Panel 2: Dynamic RSU Selection Candidates"),
        ("panel3_contact_sheet.png", panel3_items, 2, (500, 285), "Panel 3: Pose-gated V2I Fusion Candidates"),
        ("panel4_contact_sheet.png", panel4_items, 2, (420, 300), "Panel 4: HD Map Output Candidates"),
    ]:
        sheet = make_contact_sheet(items, output_root / name, columns=cols, tile=tile, title=title)
        if sheet:
            contact_sheets.append(sheet)

    manifest_path = output_root / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(report_path, output_root, manifest, contact_sheets)
    print(f"Wrote assets to {output_root}")
    print(f"Wrote manifest to {manifest_path}")
    print(f"Wrote report to {report_path}")
    for sheet in contact_sheets:
        print(f"Wrote contact sheet {sheet}")


if __name__ == "__main__":
    main()
