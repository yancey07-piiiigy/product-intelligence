"""Build a deterministic, stratified gallery for the course-sized prototype."""
import argparse
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from core import ROOT, load_catalog

DEFAULT_TARGET = 2000
DEFAULT_SEED = 6201
SUBSET_PATH = ROOT / "data/gallery-subset.json"
CURATED_IMAGES = ROOT / "data/gallery-images"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(rows, field):
    return dict(sorted(Counter(r[field] or "Unknown" for r in rows).items()))


def allocate_quotas(group_sizes, required_counts, target):
    """Largest-remainder proportional quotas, with required evaluation IDs pinned."""
    total = sum(group_sizes.values())
    desired = {k: target * n / total for k, n in group_sizes.items()}
    quotas = {k: min(group_sizes[k], max(required_counts.get(k, 0), math.floor(desired[k]))) for k in group_sizes}
    while sum(quotas.values()) < target:
        choices = [k for k in group_sizes if quotas[k] < group_sizes[k]]
        if not choices:
            break
        key = max(choices, key=lambda k: (desired[k] - quotas[k], group_sizes[k], k))
        quotas[key] += 1
    while sum(quotas.values()) > target:
        choices = [k for k in group_sizes if quotas[k] > required_counts.get(k, 0)]
        if not choices:
            raise ValueError("Target is smaller than the required evaluation gallery")
        key = max(choices, key=lambda k: (quotas[k] - desired[k], quotas[k], k))
        quotas[key] -= 1
    return quotas


def stratified_ids(rows, required_ids, target=DEFAULT_TARGET, seed=DEFAULT_SEED):
    if target < len(required_ids):
        raise ValueError("Target is smaller than the required evaluation products")
    by_id = {r["id"]: r for r in rows}
    missing = sorted(set(required_ids) - set(by_id))
    if missing:
        raise ValueError(f"Required products are unavailable: {missing[:5]}")
    if target > len(rows):
        raise ValueError(f"Target {target} exceeds {len(rows)} eligible products")

    groups = defaultdict(list)
    for row in rows:
        groups[(row["articleType"] or "Unknown", row["gender"] or "Unknown")].append(row)
    required_counts = Counter((by_id[i]["articleType"] or "Unknown", by_id[i]["gender"] or "Unknown") for i in required_ids)
    quotas = allocate_quotas({k: len(v) for k, v in groups.items()}, required_counts, target)

    selected = set(required_ids)
    for key in sorted(groups):
        candidates = [r["id"] for r in groups[key] if r["id"] not in selected]
        random.Random(f"{seed}:{key[0]}:{key[1]}").shuffle(candidates)
        selected.update(candidates[: quotas[key] - required_counts.get(key, 0)])
    if len(selected) != target:
        raise ValueError(f"Sampling produced {len(selected)} rows instead of {target}")
    return selected, {"articleType+gender": {f"{k[0]} | {k[1]}": quotas[k] for k in sorted(quotas)}}


def build_subset(target=DEFAULT_TARGET, seed=DEFAULT_SEED, source=None, copy_images=True):
    source = Path(source or ROOT / "data/images")
    protocol = json.loads((ROOT / "artifacts/protocol.json").read_text())
    queries = json.loads((ROOT / "artifacts/manifest.json").read_text())
    excluded = set(protocol["excluded_image_hashes"])
    required_ids = {q["source_id"] for q in queries if q["kind"] == "known"}
    rows, _ = load_catalog(ROOT / "data/styles.csv")

    eligible, hashes = [], {}
    for row in rows:
        path = source / f"{row['id']}.jpg"
        if not path.is_file():
            continue
        sha = file_hash(path)
        if sha in excluded:
            continue
        eligible.append(row)
        hashes[row["id"]] = sha

    selected_ids, quotas = stratified_ids(eligible, required_ids, target, seed)
    selected = [r for r in eligible if r["id"] in selected_ids]
    selected_hashes = [hashes[r["id"]] for r in selected]
    query_ids = {q["source_id"] for q in queries}

    if copy_images:
        CURATED_IMAGES.mkdir(parents=True, exist_ok=True)
        keep = selected_ids | query_ids
        for pid in sorted(keep):
            src = source / f"{pid}.jpg"
            if not src.is_file():
                raise ValueError(f"Missing source image {src}")
            dst = CURATED_IMAGES / src.name
            if not dst.exists() or file_hash(dst) != file_hash(src):
                shutil.copy2(src, dst)
        for stale in CURATED_IMAGES.glob("*.jpg"):
            if stale.stem not in keep:
                stale.unlink()

    manifest = {
        "version": "course-gallery-v1",
        "strategy": "deterministic proportional stratification by articleType and gender",
        "seed": seed,
        "target": target,
        "source_available_footwear": len(eligible) + len(excluded),
        "eligible_after_unknown_exclusion": len(eligible),
        "required_known_evaluation_products": len(required_ids),
        "selected_ids": [r["id"] for r in selected],
        "image_hashes": selected_hashes,
        "stratum_quotas": quotas,
        "selected_distribution": {k: distribution(selected, k) for k in ("articleType", "gender", "baseColour", "season", "usage")},
        "note": "The full source dataset is not searched. Known evaluation products are pinned; unknown-footwear query hashes remain excluded.",
    }
    SUBSET_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    for query in queries:
        query["original"] = f"data/gallery-images/{query['source_id']}.jpg"
    (ROOT / "artifacts/manifest.json").write_text(json.dumps(queries, indent=2, ensure_ascii=False))
    print(f"Gallery subset saved: {len(selected):,} searchable products; {len(selected_ids | query_ids):,} curated image files.")
    return manifest


def slice_existing_index():
    subset = json.loads(SUBSET_PATH.read_text())
    meta_path = ROOT / "artifacts/index.json"
    array_path = ROOT / "artifacts/image_index.npz"
    meta = json.loads(meta_path.read_text())
    old_positions = {row["id"]: i for i, row in enumerate(meta["rows"])}
    missing = [pid for pid in subset["selected_ids"] if pid not in old_positions]
    if missing:
        raise ValueError(f"Existing index cannot be sliced; missing IDs: {missing[:5]}")
    positions = np.array([old_positions[pid] for pid in subset["selected_ids"]], dtype=int)
    with np.load(array_path, allow_pickle=False) as arrays:
        clip = arrays["clip"][positions]
        hist = arrays["hist"][positions]
    old_rows = {r["id"]: r for r in meta["rows"]}
    old_hashes = {r["id"]: h for r, h in zip(meta["rows"], meta["image_hashes"])}
    meta["rows"] = [old_rows[pid] for pid in subset["selected_ids"]]
    meta["image_hashes"] = [old_hashes[pid] for pid in subset["selected_ids"]]
    meta["gallery_subset"] = str(SUBSET_PATH.relative_to(ROOT))
    meta["gallery_target"] = subset["target"]
    meta["signature"] = hashlib.sha256(json.dumps({k: meta[k] for k in ("model", "model_sha256", "preprocess", "rows", "image_hashes", "protocol_hash")}, sort_keys=True).encode()).hexdigest()
    np.savez_compressed(array_path, clip=clip, hist=hist)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False))
    print(f"Existing CLIP index reduced to {len(meta['rows']):,} rows. Rebuild geometry and rerun benchmark next.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--source", default=str(ROOT / "data/images"))
    parser.add_argument("--slice-existing-index", action="store_true")
    args = parser.parse_args()
    build_subset(args.target, args.seed, args.source)
    if args.slice_existing_index:
        slice_existing_index()


if __name__ == "__main__":
    main()
