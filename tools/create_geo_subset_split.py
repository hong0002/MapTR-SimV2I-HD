#!/usr/bin/env python
"""Create a geographically separated SimV2I-HD MapTR split.

This utility merges the existing 20k train/val/test PKLs into one candidate
pool and assigns a new split from ego world positions only. It does not train a
model and does not modify the source dataset.

Default policy:
    test: ego_y < -90
    val:  ego_y > 55
    train: samples at least 30 m away from both held-out y bands

Example:
    python tools/create_geo_subset_split.py \
      --input-root "${SIMV2I_HD_ROOT}" \
      --output-root "${SIMV2I_HD_GEO_ROOT}"
"""

import argparse
import copy
import csv
import json
import math
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SPLITS = ("train", "val", "test")
MAP_CLASSES_FALLBACK = ("divider", "ped_crossing", "boundary")
MODEL_MAP_CLASSES = ("divider", "boundary", "ped_crossing")
ROI_LOCAL_CORNERS = ((-15.0, -30.0), (15.0, -30.0), (15.0, 30.0), (-15.0, 30.0))


def parse_args():
    default_input = os.environ.get(
        "SIMV2I_HD_ROOT",
        "data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k",
    )
    default_output = os.environ.get("SIMV2I_HD_GEO_ROOT")
    if default_output is None:
        input_path = Path(default_input)
        if os.environ.get("SIMV2I_HD_ROOT"):
            default_output = str(
                input_path.parent / "simv2i_hd_benchmark_v2_dynamic_rsu_20k_geo_subset"
            )
        else:
            default_output = "data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k_geo_subset"

    parser = argparse.ArgumentParser(
        description="Create a geographic SimV2I-HD MapTR subset split."
    )
    parser.add_argument("--input-root", default=default_input, type=Path)
    parser.add_argument("--output-root", default=default_output, type=Path)
    parser.add_argument(
        "--image-root",
        default=os.environ.get("SIMV2I_HD_IMAGE_ROOT")
        or os.environ.get("SIMV2I_HD_ROOT")
        or ".",
        type=Path,
        help="Root used to validate relative image paths.",
    )
    parser.add_argument("--test-y-max", type=float, default=-90.0)
    parser.add_argument("--val-y-min", type=float, default=55.0)
    parser.add_argument("--buffer-m", type=float, default=30.0)
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--skip-roi-overlap", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_payload(path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "infos" in payload:
        infos = list(payload["infos"])
        metadata = dict(payload.get("metadata", {}))
    elif isinstance(payload, list):
        infos = list(payload)
        metadata = {}
    else:
        raise TypeError("{} must be a list or a dict with 'infos'.".format(path))
    return infos, metadata


def sample_token(info, fallback_index):
    return str(info.get("token") or info.get("sample_idx") or "sample_{:06d}".format(fallback_index))


def sample_xy(info):
    if "ego2global_translation" in info:
        value = info["ego2global_translation"]
    elif "can_bus" in info and len(info["can_bus"]) >= 2:
        value = info["can_bus"]
    else:
        raise KeyError("Sample {} has no ego world translation.".format(info.get("token")))
    return float(value[0]), float(value[1])


def sample_yaw(info):
    if "ego2global_rotation" in info:
        q = [float(v) for v in info["ego2global_rotation"]]
        if len(q) == 4:
            w, x, y, z = q
            return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    if "can_bus" in info and len(info["can_bus"]) >= 17:
        return float(info["can_bus"][-2])
    return 0.0


def split_payload(infos, metadata, split_name, report_meta):
    payload_meta = dict(metadata)
    payload_meta.update(
        {
            "split": split_name,
            "run_id": report_meta["dataset_name"],
            "geo_subset": True,
            "geo_subset_policy": report_meta["policy"],
            "source_dataset": report_meta["source_dataset"],
            "source_total_count": report_meta["source_total_count"],
            "map_classes": report_meta["source_map_classes"],
            "model_map_classes": list(MODEL_MAP_CLASSES),
        }
    )
    return {"infos": infos, "metadata": payload_meta}


def classify_geo_split(y, test_y_max, val_y_min, buffer_m):
    if y < test_y_max:
        return "test"
    if y > val_y_min:
        return "val"
    if y > test_y_max + buffer_m and y < val_y_min - buffer_m:
        return "train"
    return "ignore"


def nearest_distances(query, ref):
    if len(query) == 0 or len(ref) == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(ref).query(query, k=1)
        return np.asarray(distances), np.asarray(indices, dtype=np.int64)
    except Exception:
        distances = np.empty((len(query),), dtype=np.float64)
        indices = np.empty((len(query),), dtype=np.int64)
        chunk = 512
        for start in range(0, len(query), chunk):
            q = query[start : start + chunk]
            diff = q[:, None, :] - ref[None, :, :]
            dist2 = np.sum(diff * diff, axis=2)
            idx = np.argmin(dist2, axis=1)
            distances[start : start + chunk] = np.sqrt(dist2[np.arange(len(q)), idx])
            indices[start : start + chunk] = idx
        return distances, indices


def leakage_summary(distances):
    if len(distances) == 0:
        return {
            "count": 0,
            "within_1m": 0,
            "within_2m": 0,
            "within_5m": 0,
            "within_10m": 0,
            "within_1m_pct": 0.0,
            "within_2m_pct": 0.0,
            "within_5m_pct": 0.0,
            "within_10m_pct": 0.0,
            "min": None,
            "mean": None,
            "median": None,
            "max": None,
        }
    result = {
        "count": int(len(distances)),
        "min": float(np.min(distances)),
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "max": float(np.max(distances)),
    }
    for threshold in (1, 2, 5, 10):
        count = int(np.sum(distances <= threshold))
        result["within_{}m".format(threshold)] = count
        result["within_{}m_pct".format(threshold)] = float(count * 100.0 / len(distances))
    return result


def write_nearest_csv(path, query_records, train_records, distances, indices):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_token",
                "geo_split",
                "original_split",
                "ego_x",
                "ego_y",
                "nearest_train_distance_m",
                "nearest_train_token",
                "nearest_train_original_split",
                "nearest_train_x",
                "nearest_train_y",
            ],
        )
        writer.writeheader()
        for record, distance, nearest_idx in zip(query_records, distances, indices):
            nearest = train_records[int(nearest_idx)]
            writer.writerow(
                {
                    "sample_token": record["token"],
                    "geo_split": record["geo_split"],
                    "original_split": record["original_split"],
                    "ego_x": "{:.6f}".format(record["x"]),
                    "ego_y": "{:.6f}".format(record["y"]),
                    "nearest_train_distance_m": "{:.6f}".format(float(distance)),
                    "nearest_train_token": nearest["token"],
                    "nearest_train_original_split": nearest["original_split"],
                    "nearest_train_x": "{:.6f}".format(nearest["x"]),
                    "nearest_train_y": "{:.6f}".format(nearest["y"]),
                }
            )


def class_distribution(records, source_map_classes):
    counter = Counter()
    for record in records:
        info = record["info"]
        labels = info.get("maptr_gt_labels") or info.get("gt_labels") or []
        for label in labels:
            label_int = int(label)
            if 0 <= label_int < len(source_map_classes):
                counter[source_map_classes[label_int]] += 1
            else:
                counter["label_{}".format(label_int)] += 1
    return dict(sorted(counter.items()))


def original_split_composition(records):
    return dict(sorted(Counter(record["original_split"] for record in records).items()))


def rsu_distribution(records):
    counter = Counter()
    for record in records:
        for rsu_id in (record["info"].get("rsu_cams") or {}).keys():
            counter[rsu_id] += 1
    return dict(sorted(counter.items()))


def resolve_path(root, value):
    path = Path(os.fspath(value))
    if path.is_absolute():
        return path
    return root / path


def image_path_report(records, image_root):
    total = 0
    missing = []
    for record in records:
        info = record["info"]
        for group_name in ("cams", "rsu_cams"):
            cams = info.get(group_name) or {}
            for camera_name, camera_info in cams.items():
                if not isinstance(camera_info, dict):
                    continue
                data_path = camera_info.get("data_path") or camera_info.get("img_path")
                if not data_path:
                    continue
                total += 1
                resolved = resolve_path(image_root, data_path)
                if not resolved.is_file() and len(missing) < 50:
                    missing.append(
                        {
                            "sample_token": record["token"],
                            "camera": camera_name,
                            "path": data_path,
                            "resolved_path": str(resolved),
                        }
                    )
    return {"checked_paths": total, "missing_count_limited": len(missing), "first_missing": missing}


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_map_gt(path, records, source_map_classes):
    gts = []
    for record in records:
        info = record["info"]
        vectors = []
        labels = info.get("maptr_gt_labels") or info.get("gt_labels") or []
        points = info.get("maptr_gt_fixed_points") or info.get("gt_vecs") or []
        for pts, label in zip(points, labels):
            label_int = int(label)
            cls_name = (
                source_map_classes[label_int]
                if 0 <= label_int < len(source_map_classes)
                else "label_{}".format(label_int)
            )
            pts_json = jsonable(pts)
            vectors.append(
                {
                    "pts": pts_json,
                    "pts_num": len(pts_json),
                    "cls_name": cls_name,
                    "type": label_int,
                }
            )
        gts.append({"sample_token": record["token"], "vectors": vectors})
    payload = {
        "GTs": gts,
        "meta": {
            "map_classes": list(source_map_classes),
            "geo_subset": True,
            "count": len(gts),
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def roi_polygon(record):
    try:
        from shapely.geometry import Polygon
    except Exception:
        return None
    x = record["x"]
    y = record["y"]
    yaw = record["yaw"]
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    world = []
    for local_x, local_y in ROI_LOCAL_CORNERS:
        world.append(
            (
                x + cos_yaw * local_x - sin_yaw * local_y,
                y + sin_yaw * local_x + cos_yaw * local_y,
            )
        )
    return Polygon(world)


def roi_overlap_summary(query_records, train_records):
    try:
        from shapely.strtree import STRtree
    except Exception as exc:
        return {"available": False, "reason": repr(exc)}

    train_polygons = [roi_polygon(record) for record in train_records]
    tree = STRtree(train_polygons)
    overlap_count = 0
    overlap_areas = []
    for record in query_records:
        polygon = roi_polygon(record)
        has_overlap = False
        max_area = 0.0
        for candidate in tree.query(polygon):
            if not polygon.intersects(candidate):
                continue
            area = polygon.intersection(candidate).area
            if area > 1e-6:
                has_overlap = True
                max_area = max(max_area, float(area))
        if has_overlap:
            overlap_count += 1
            overlap_areas.append(max_area)

    result = {
        "available": True,
        "count": len(query_records),
        "overlap_count": int(overlap_count),
        "overlap_pct": float(overlap_count * 100.0 / len(query_records))
        if query_records
        else 0.0,
    }
    if overlap_areas:
        result.update(
            {
                "max_overlap_area_min": float(np.min(overlap_areas)),
                "max_overlap_area_mean": float(np.mean(overlap_areas)),
                "max_overlap_area_median": float(np.median(overlap_areas)),
                "max_overlap_area_max": float(np.max(overlap_areas)),
            }
        )
    return result


def make_markdown(report):
    lines = [
        "# SimV2I-HD Geo Subset Split Report",
        "",
        "## Construction",
        "",
        "- Source split PKLs were merged into one 20k candidate pool.",
        "- The original split label is stored only as `original_split` for analysis.",
        "- New split assignment uses ego world y-position only.",
        "- Test region: `ego_y < {}`.".format(report["policy"]["test_y_max"]),
        "- Validation region: `ego_y > {}`.".format(report["policy"]["val_y_min"]),
        "- Train region excludes a `{}` m buffer around held-out y bands.".format(
            report["policy"]["buffer_m"]
        ),
        "- Boundary samples outside train/val/test are removed into `ignore`.",
        "",
        "## Counts",
        "",
    ]
    for split_name in ("train", "val", "test", "ignore"):
        lines.append("- {}: {}".format(split_name, report["counts"][split_name]))
    lines.extend(
        [
            "",
            "## Leakage Summary",
            "",
            "| Split | <=1m | <=2m | <=5m | <=10m | min | mean | median | max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in ("original_val_vs_train", "original_test_vs_train", "geo_val_vs_train", "geo_test_vs_train"):
        item = report["leakage"][key]
        lines.append(
            "| {} | {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                key,
                item["within_1m"],
                item["within_2m"],
                item["within_5m"],
                item["within_10m"],
                item["min"],
                item["mean"],
                item["median"],
                item["max"],
            )
        )
    lines.extend(["", "## Original Split Composition", ""])
    for split_name in ("train", "val", "test", "ignore"):
        lines.append(
            "- {}: {}".format(
                split_name,
                json.dumps(report["original_split_composition"][split_name], sort_keys=True),
            )
        )
    lines.extend(
        [
            "",
            "## ROI Overlap",
            "",
            "ROI overlap is computed from the BEV map window x=[-15,15], y=[-30,30] when Shapely is available.",
            "",
            "```json",
            json.dumps(report["roi_overlap"], indent=2, sort_keys=True),
            "```",
            "",
            "## Outputs",
            "",
        ]
    )
    for key, value in report["outputs"].items():
        lines.append("- {}: `{}`".format(key, value))
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    input_root = args.input_root
    output_root = args.output_root

    if input_root.resolve() == output_root.resolve():
        raise ValueError("output-root must differ from input-root.")

    source_infos = {}
    source_metadata = {}
    source_counts = {}
    for split_name in SPLITS:
        path = input_root / "simv2i_maptr_infos_{}.pkl".format(split_name)
        if not path.is_file():
            raise FileNotFoundError(path)
        infos, metadata = load_payload(path)
        source_infos[split_name] = infos
        source_metadata[split_name] = metadata
        source_counts[split_name] = len(infos)

    source_map_classes = (
        source_metadata.get("train", {}).get("map_classes")
        or source_metadata.get("val", {}).get("map_classes")
        or source_metadata.get("test", {}).get("map_classes")
        or list(MAP_CLASSES_FALLBACK)
    )

    records = []
    seen_tokens = set()
    for split_name in SPLITS:
        for local_index, info in enumerate(source_infos[split_name]):
            token = sample_token(info, len(records))
            if token in seen_tokens:
                raise ValueError("Duplicate sample token: {}".format(token))
            seen_tokens.add(token)
            x, y = sample_xy(info)
            geo_split = classify_geo_split(
                y, args.test_y_max, args.val_y_min, args.buffer_m
            )
            records.append(
                {
                    "token": token,
                    "original_split": split_name,
                    "original_local_index": local_index,
                    "geo_split": geo_split,
                    "x": x,
                    "y": y,
                    "yaw": sample_yaw(info),
                    "info": info,
                }
            )

    grouped = {name: [] for name in ("train", "val", "test", "ignore")}
    for record in records:
        grouped[record["geo_split"]].append(record)

    for split_name in grouped:
        grouped[split_name].sort(
            key=lambda item: (
                item["info"].get("timestamp", 0),
                item["original_split"],
                item["token"],
            )
        )

    target_paths = [
        output_root / "simv2i_maptr_infos_train.pkl",
        output_root / "simv2i_maptr_infos_val.pkl",
        output_root / "simv2i_maptr_infos_test.pkl",
        output_root / "simv2i_maptr_map_gt_train.json",
        output_root / "simv2i_maptr_map_gt_val.json",
        output_root / "simv2i_maptr_map_gt_test.json",
        output_root / "split_geo_subset_report.json",
        output_root / "split_geo_subset_report.md",
        output_root / "nearest_train_distance_test.csv",
        output_root / "nearest_train_distance_val.csv",
    ]
    if not args.overwrite:
        existing = [str(path) for path in target_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing files. Use --overwrite if intended: {}".format(
                    ", ".join(existing)
                )
            )
    output_root.mkdir(parents=True, exist_ok=True)

    train_points = np.array([[r["x"], r["y"]] for r in grouped["train"]], dtype=np.float64)
    val_points = np.array([[r["x"], r["y"]] for r in grouped["val"]], dtype=np.float64)
    test_points = np.array([[r["x"], r["y"]] for r in grouped["test"]], dtype=np.float64)

    original_train = [r for r in records if r["original_split"] == "train"]
    original_val = [r for r in records if r["original_split"] == "val"]
    original_test = [r for r in records if r["original_split"] == "test"]
    original_train_points = np.array([[r["x"], r["y"]] for r in original_train], dtype=np.float64)
    original_val_points = np.array([[r["x"], r["y"]] for r in original_val], dtype=np.float64)
    original_test_points = np.array([[r["x"], r["y"]] for r in original_test], dtype=np.float64)

    val_dist, val_idx = nearest_distances(val_points, train_points)
    test_dist, test_idx = nearest_distances(test_points, train_points)
    orig_val_dist, _ = nearest_distances(original_val_points, original_train_points)
    orig_test_dist, _ = nearest_distances(original_test_points, original_train_points)

    if not args.skip_image_check:
        image_report = image_path_report(
            grouped["train"] + grouped["val"] + grouped["test"], args.image_root
        )
    else:
        image_report = {"skipped": True}

    roi_overlap = {}
    if args.skip_roi_overlap:
        roi_overlap = {"skipped": True}
    else:
        roi_overlap["val_vs_train"] = roi_overlap_summary(grouped["val"], grouped["train"])
        roi_overlap["test_vs_train"] = roi_overlap_summary(grouped["test"], grouped["train"])

    policy = {
        "method": "manual_y_band_with_buffer",
        "test_y_max": args.test_y_max,
        "val_y_min": args.val_y_min,
        "buffer_m": args.buffer_m,
        "train_condition": "test_y_max + buffer_m < ego_y < val_y_min - buffer_m",
    }
    dataset_name = "simv2i_hd_benchmark_v2_dynamic_rsu_20k_geo_subset"
    report_meta = {
        "dataset_name": dataset_name,
        "source_dataset": "simv2i_hd_benchmark_v2_dynamic_rsu_20k",
        "source_total_count": len(records),
        "source_map_classes": list(source_map_classes),
        "policy": policy,
    }

    outputs = {}
    for split_name in SPLITS:
        out_infos = []
        for record in grouped[split_name]:
            info_copy = copy.deepcopy(record["info"])
            info_copy["original_split"] = record["original_split"]
            info_copy["geo_split"] = split_name
            info_copy["geo_split_policy"] = policy["method"]
            out_infos.append(info_copy)
            record["info"] = info_copy

        payload = split_payload(
            out_infos, source_metadata.get("train", {}), split_name, report_meta
        )
        pkl_path = output_root / "simv2i_maptr_infos_{}.pkl".format(split_name)
        with pkl_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=4)
        outputs["{}_pkl".format(split_name)] = str(pkl_path)

        gt_path = output_root / "simv2i_maptr_map_gt_{}.json".format(split_name)
        write_map_gt(gt_path, grouped[split_name], source_map_classes)
        outputs["{}_map_gt_json".format(split_name)] = str(gt_path)

    test_csv = output_root / "nearest_train_distance_test.csv"
    val_csv = output_root / "nearest_train_distance_val.csv"
    write_nearest_csv(test_csv, grouped["test"], grouped["train"], test_dist, test_idx)
    write_nearest_csv(val_csv, grouped["val"], grouped["train"], val_dist, val_idx)
    outputs["nearest_train_distance_test_csv"] = str(test_csv)
    outputs["nearest_train_distance_val_csv"] = str(val_csv)

    report = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "image_root": str(args.image_root),
        "source_counts": source_counts,
        "source_total_count": len(records),
        "source_map_classes": list(source_map_classes),
        "model_map_classes": list(MODEL_MAP_CLASSES),
        "policy": policy,
        "counts": {name: len(grouped[name]) for name in ("train", "val", "test", "ignore")},
        "removed_boundary_samples": len(grouped["ignore"]),
        "leakage": {
            "original_val_vs_train": leakage_summary(orig_val_dist),
            "original_test_vs_train": leakage_summary(orig_test_dist),
            "geo_val_vs_train": leakage_summary(val_dist),
            "geo_test_vs_train": leakage_summary(test_dist),
        },
        "original_split_composition": {
            name: original_split_composition(grouped[name])
            for name in ("train", "val", "test", "ignore")
        },
        "class_distribution": {
            name: class_distribution(grouped[name], source_map_classes)
            for name in ("train", "val", "test")
        },
        "selected_rsu_distribution": {
            name: rsu_distribution(grouped[name]) for name in ("train", "val", "test")
        },
        "image_path_report": image_report,
        "roi_overlap": roi_overlap,
        "outputs": outputs,
    }

    report_json = output_root / "split_geo_subset_report.json"
    report_md = output_root / "split_geo_subset_report.md"
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report_md.write_text(make_markdown(report))
    outputs["report_json"] = str(report_json)
    outputs["report_md"] = str(report_md)

    print("Geo subset split written to {}".format(output_root))
    print("Counts:", report["counts"])
    print("Geo test leakage:", report["leakage"]["geo_test_vs_train"])
    print("Report:", report_md)


if __name__ == "__main__":
    main()
