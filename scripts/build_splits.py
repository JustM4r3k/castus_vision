"""Merge imported datasets, detect duplicates, create leakage-safe splits and write reports.

    python scripts/build_splits.py
    python scripts/build_splits.py --inputs datasets/internal/taco.coco.json datasets/internal/bepli.coco.json

Outputs in --out-dir: merged.coco.json, hashes.json (cache), splits.json,
dedup_report.json, composition_report.json / .md.
If splits.json exists its assignments are kept (frozen); use --reset-splits to re-split from scratch.
"""
import argparse
import hashlib
import json
from pathlib import Path

from castus_vision.coco_io import save_json, sha256_file
from castus_vision.data.dedup import cluster_images, compute_hashes
from castus_vision.data.merge import merge_cocos
from castus_vision.data.report import composition_report, report_markdown
from castus_vision.data.split import assign_splits, check_leakage


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", type=Path,
                   default=[Path("datasets/internal/taco.coco.json"), Path("datasets/internal/bepli.coco.json")])
    p.add_argument("--out-dir", type=Path, default=Path("datasets/merged"))
    p.add_argument("--hash-threshold", type=int, default=10, help="max dHash Hamming distance (of 256 bits)")
    p.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--imgsz", type=int, default=640, help="resolution used for object-size bins in the report")
    p.add_argument("--reset-splits", action="store_true")
    a = p.parse_args()

    out = a.out_dir
    coco = merge_cocos(a.inputs)
    merged_path = out / "merged.coco.json"
    save_json(coco, merged_path)

    print("Hashing images (cached after the first run)...")
    hashes = compute_hashes(coco, out / "hashes.json")
    groups, dedup = cluster_images(hashes, a.hash_threshold)
    save_json(dedup, out / "dedup_report.json")

    splits_path = out / "splits.json"
    existing = None
    if splits_path.is_file() and not a.reset_splits:
        existing = {k: v["split"] for k, v in json.loads(splits_path.read_text())["images"].items()}
    ratios = dict(zip(("train", "val", "test"), a.ratios))
    assignment, excluded, conflicts = assign_splits(coco, groups, ratios, a.seed, existing)
    leak = check_leakage(assignment, groups, hashes)
    if any(leak.values()):
        raise SystemExit(f"Leakage detected: {leak}")

    # 'info' holds machine-specific absolute paths; excluding it keeps the version identical across machines.
    content = {k: v for k, v in coco.items() if k != "info"}
    version = hashlib.sha256((json.dumps(content, sort_keys=True) + json.dumps(assignment, sort_keys=True)).encode()
                             ).hexdigest()[:16]
    save_json({
        "dataset_version": version, "seed": a.seed, "ratios": ratios,
        "hash_threshold": a.hash_threshold, "merged_coco_sha256": sha256_file(merged_path),
        "images": {k: {"split": s, "group_id": groups[k]} for k, s in sorted(assignment.items())},
        "excluded": excluded, "conflicts": conflicts, "leakage_check": leak,
    }, splits_path)

    report = composition_report(coco, assignment, excluded, a.imgsz)
    report["dataset_version"] = version
    save_json(report, out / "composition_report.json")
    (out / "composition_report.md").write_text(report_markdown(report, {"Duplicate detection": json.dumps(
        {k: v for k, v in dedup.items() if k != "largest_groups"}, indent=1)}), encoding="utf-8")
    print(report_markdown(report))
    print(f"dataset_version={version}  groups={dedup['groups']}  conflicts={len(conflicts)}  leakage={leak}")
    print(f"Wrote reports to {out}")


if __name__ == "__main__":
    main()
