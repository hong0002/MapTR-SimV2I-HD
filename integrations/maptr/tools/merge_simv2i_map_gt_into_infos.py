import argparse
import json
import pickle
import math
from pathlib import Path
from collections import Counter

CLASSES = {"divider", "boundary", "ped_crossing"}

def valid_point(p):
    return (
        isinstance(p, (list, tuple))
        and len(p) >= 2
        and isinstance(p[0], (int, float))
        and isinstance(p[1], (int, float))
        and math.isfinite(float(p[0]))
        and math.isfinite(float(p[1]))
    )

def normalize_vector(v):
    cls_name = v.get("cls_name")
    if cls_name not in CLASSES:
        return None

    pts = v.get("pts")
    if not isinstance(pts, list):
        return None

    clean_pts = []
    for p in pts:
        if valid_point(p):
            clean_pts.append([float(p[0]), float(p[1])])

    if len(clean_pts) < 2:
        return None

    return {
        "pts": clean_pts,
        "cls_name": cls_name,
    }

def load_infos(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "infos" in data:
        return data, data["infos"]

    if isinstance(data, list):
        return data, data

    raise TypeError(f"Unsupported pickle top-level type: {type(data)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/maptr/simv2i_hd_v1")
    parser.add_argument("--backup", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    total_summary = {}

    for split in ["train", "val", "test"]:
        pkl_path = dataset_dir / f"simv2i_maptr_infos_{split}.pkl"
        json_path = dataset_dir / f"simv2i_maptr_map_gt_{split}.json"

        print(f"\n[split] {split}")
        print(f"  pkl : {pkl_path}")
        print(f"  json: {json_path}")

        if not pkl_path.exists():
            raise FileNotFoundError(pkl_path)
        if not json_path.exists():
            raise FileNotFoundError(json_path)

        data, infos = load_infos(pkl_path)

        gt_obj = json.loads(json_path.read_text())
        gts = gt_obj.get("GTs")
        if not isinstance(gts, list):
            raise KeyError(f"{json_path} does not contain list field 'GTs'")

        token_to_vectors = {}
        for item in gts:
            token = item.get("sample_token")
            vectors = item.get("vectors", [])
            if token is None:
                continue
            token_to_vectors[token] = vectors

        missing_tokens = []
        invalid_vectors = 0
        class_counter = Counter()
        instance_count = 0

        for info in infos:
            token = info.get("token")
            if token not in token_to_vectors:
                missing_tokens.append(token)
                info["gt_vectors"] = []
                continue

            normalized = []
            for v in token_to_vectors[token]:
                nv = normalize_vector(v)
                if nv is None:
                    invalid_vectors += 1
                    continue
                normalized.append(nv)
                class_counter[nv["cls_name"]] += 1

            info["gt_vectors"] = normalized
            instance_count += len(normalized)

        if args.backup:
            backup = pkl_path.with_suffix(pkl_path.suffix + ".before_gt_merge.bak")
            if not backup.exists():
                backup.write_bytes(pkl_path.read_bytes())
                print(f"  backup -> {backup}")
            else:
                print(f"  backup already exists: {backup}")

        with open(pkl_path, "wb") as f:
            pickle.dump(data, f, protocol=4)

        print(f"  infos          : {len(infos)}")
        print(f"  json GTs       : {len(gts)}")
        print(f"  missing tokens : {len(missing_tokens)}")
        print(f"  invalid vectors: {invalid_vectors}")
        print(f"  GT instances   : {instance_count}")
        print(f"  class counts   : {dict(class_counter)}")
        print(f"  saved protocol : 4")

        total_summary[split] = {
            "infos": len(infos),
            "json_gts": len(gts),
            "missing_tokens": len(missing_tokens),
            "invalid_vectors": invalid_vectors,
            "gt_instances": instance_count,
            "class_counts": dict(class_counter),
        }

        if missing_tokens:
            print("  first missing tokens:", missing_tokens[:5])

    print("\n[summary]")
    print(json.dumps(total_summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
