#!/usr/bin/env python
"""Select qualitative SimV2I-HD MapTR samples with simple proxy scores."""

import argparse
import os
from collections import Counter

from visualize_simv2i_maptr_predictions import (
    CLASS_NAMES,
    load_gt_records,
    load_prediction_records,
    vectors_for_item,
)


DEFAULT_PREDS = {
    "ego_only": "outputs/maptr/visualization_predictions/ego_only_test_epoch18.pkl",
    "dynamic_top4": (
        "outputs/maptr/visualization_predictions/dynamic_top4_test_epoch18.pkl"
    ),
    "constant_gate": (
        "outputs/maptr/visualization_predictions/constant_gate_test_epoch18.pkl"
    ),
    "pose_gated_top4": (
        "outputs/maptr/visualization_predictions/pose_gated_top4_test_epoch18.pkl"
    ),
}
DEFAULT_DATA_ROOT = os.environ.get(
    "SIMV2I_HD_ROOT",
    "data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank qualitative samples using count/score proxy metrics."
    )
    parser.add_argument(
        "--gt-json",
        default=os.path.join(DEFAULT_DATA_ROOT, "simv2i_maptr_map_gt_test.json"),
    )
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--pred-pkl",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help="Override/add prediction file.",
    )
    return parser.parse_args()


def parse_pred_specs(specs):
    parsed = dict(DEFAULT_PREDS)
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--pred-pkl must be MODEL=PATH, got: {spec}")
        name, path = spec.split("=", 1)
        parsed[name] = path
    return parsed


def count_by_class(vectors):
    return Counter(vec["cls_name"] for vec in vectors if vec["cls_name"] in CLASS_NAMES)


def count_error(pred_counts, gt_counts):
    return sum(abs(pred_counts.get(cls, 0) - gt_counts.get(cls, 0)) for cls in CLASS_NAMES)


def mean_score(vectors):
    if not vectors:
        return 0.0
    return sum(vec["score"] for vec in vectors) / len(vectors)


def score_sample(gt_item, pred_item, score_thr):
    gt_vectors = vectors_for_item(gt_item, score_thr=None)
    pred_vectors = vectors_for_item(pred_item, score_thr=score_thr)
    return {
        "count_error": count_error(count_by_class(pred_vectors), count_by_class(gt_vectors)),
        "mean_score": mean_score(pred_vectors),
        "pred_count": len(pred_vectors),
        "gt_count": len(gt_vectors),
    }


def print_section(title, rows, top_k):
    print(f"\n{title}")
    for row in rows[:top_k]:
        print(
            f"idx={row['index']:04d} token={row['token']} score={row['score']:.3f} "
            f"pose_err={row.get('pose_err', 0)} dyn_err={row.get('dyn_err', 0)} "
            f"ego_err={row.get('ego_err', 0)} pose_count={row.get('pose_count', 0)}"
        )


def main():
    args = parse_args()
    _, gt_records, _ = load_gt_records(args.gt_json)
    specs = parse_pred_specs(args.pred_pkl)
    predictions = {}
    for model_name, path in specs.items():
        _, _, token_to_pred, _ = load_prediction_records(path, gt_records)
        predictions[model_name] = token_to_pred

    rows = []
    for index, gt_item in enumerate(gt_records):
        token = gt_item.get("sample_token") or str(index)
        model_scores = {}
        for model_name, token_to_pred in predictions.items():
            model_scores[model_name] = score_sample(
                gt_item, token_to_pred.get(token, {"vectors": []}), args.score_thr
            )
        pose = model_scores.get("pose_gated_top4", {})
        dyn = model_scores.get("dynamic_top4", {})
        ego = model_scores.get("ego_only", {})
        rows.append(
            {
                "index": index,
                "token": token,
                "pose_err": pose.get("count_error", 0),
                "dyn_err": dyn.get("count_error", 0),
                "ego_err": ego.get("count_error", 0),
                "pose_count": pose.get("pred_count", 0),
                "score": 0.0,
            }
        )

    pose_better = []
    dynamic_failed = []
    ego_pose_gap = []
    for row in rows:
        better = row["dyn_err"] - row["pose_err"]
        pose_better.append(dict(row, score=better))
        dynamic_failed.append(dict(row, score=row["dyn_err"]))
        ego_pose_gap.append(dict(row, score=abs(row["ego_err"] - row["pose_err"])))

    print_section(
        "Pose-gated better than Dynamic Top-4 candidates",
        sorted(pose_better, key=lambda item: item["score"], reverse=True),
        args.top_k,
    )
    print_section(
        "Dynamic Top-4 failure candidates",
        sorted(dynamic_failed, key=lambda item: item["score"], reverse=True),
        args.top_k,
    )
    print_section(
        "Large Ego-only vs Pose-gated difference candidates",
        sorted(ego_pose_gap, key=lambda item: item["score"], reverse=True),
        args.top_k,
    )
    print(
        "\nNote: this is a count/score proxy ranking, not a per-sample AP or "
        "Chamfer metric."
    )


if __name__ == "__main__":
    main()
