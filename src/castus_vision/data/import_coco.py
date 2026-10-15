"""Import a COCO-format source dataset into the internal CASTUS COCO format.

Every source label is looked up in an explicit mapping file. Labels mapped to
"ignore" or "review" are never turned into training labels, and the image that
contained them is flagged `has_unverified_objects` so it is not later treated as a
genuine negative.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from castus_vision.annotations import AnnotationError, normalize_annotation
from castus_vision.coco_io import load_coco, sha256_file, upright_size
from castus_vision.taxonomy import IGNORE, REVIEW, load_mapping, load_taxonomy, DEFAULT_TAXONOMY

SCALE_TOLERANCE = 0.01  # max relative difference between x and y scale factors


def import_coco(dataset: str, source_json: Path, image_root: Path, mapping_path: Path,
                taxonomy_path: Path = DEFAULT_TAXONOMY, check_images: bool = True):
    """Return (internal_coco, report)."""
    taxonomy = load_taxonomy(taxonomy_path)
    name_to_id = {n: i for i, n in taxonomy.items()}
    mapping = load_mapping(mapping_path, taxonomy)
    src = load_coco(source_json)

    src_cats = {c["id"]: c["name"] for c in src["categories"]}
    unmapped = sorted(n for n in src_cats.values() if n not in mapping)
    if unmapped:
        raise ValueError(f"Source labels missing from mapping {mapping_path}: {unmapped}")
    unused_mapping = sorted(set(mapping) - set(src_cats.values()))

    anns_by_image: dict[int, list] = {}
    for a in src["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    images_out, anns_out = [], []
    rejected_images: Counter = Counter()
    rejected_anns: Counter = Counter()
    label_stats = {n: {"action": mapping[n], "instances": 0} for n in src_cats.values()}
    kept_per_class: Counter = Counter()
    n_rescaled = n_bbox_mismatch = 0

    for img in src["images"]:
        path = Path(image_root) / img["file_name"]
        w, h = img["width"], img["height"]
        sx = sy = 1.0
        rescaled = False
        if check_images:
            if not path.is_file():
                rejected_images["missing_file"] += 1
                continue
            try:
                aw, ah = upright_size(path)
            except Exception:
                rejected_images["unreadable_file"] += 1
                continue
            if (aw, ah) != (w, h):
                sx, sy = aw / w, ah / h
                if abs(sx - sy) / sx > SCALE_TOLERANCE:
                    rejected_images["size_mismatch_not_uniform"] += 1
                    continue
                w, h, rescaled = aw, ah, True
                n_rescaled += 1

        new_id = len(images_out) + 1
        unverified = invalid = False
        kept_here = []
        for a in anns_by_image.get(img["id"], []):
            label = src_cats[a["category_id"]]
            label_stats[label]["instances"] += 1
            action = mapping[label]
            if action in (IGNORE, REVIEW):
                unverified = True
                continue
            if a.get("iscrowd", 0):
                rejected_anns["iscrowd"] += 1
                invalid = True
                continue
            try:
                norm = normalize_annotation(a.get("segmentation"), a.get("bbox"), w, h, sx, sy)
            except AnnotationError as e:
                rejected_anns[e.reason] += 1
                invalid = True
                continue
            n_bbox_mismatch += norm.pop("bbox_mismatch")
            kept_here.append({
                "image_id": new_id,
                "category_id": name_to_id[action],
                "iscrowd": 0,
                "source_category": label,
                "source_annotation_id": a["id"],
                **norm,
            })
            kept_per_class[action] += 1

        images_out.append({
            "id": new_id,
            "file_name": img["file_name"],
            "width": w,
            "height": h,
            "source": dataset,
            "source_image_id": img["id"],
            "group_id": f"{dataset}:{img['id']}",  # placeholder until dedup/grouping assigns real groups
            "rescaled_from": [img["width"], img["height"]] if rescaled else None,
            "has_unverified_objects": unverified or invalid,
        })
        anns_out.extend(kept_here)

    for i, a in enumerate(anns_out, start=1):
        a["id"] = i

    coco = {
        "info": {
            "dataset": dataset,
            "source_json": str(source_json),
            "source_json_sha256": sha256_file(source_json),
            "mapping_file": str(mapping_path),
            "mapping_sha256": sha256_file(mapping_path),
            "taxonomy_version": 1,
            "image_roots": {dataset: str(image_root)},
        },
        "categories": [{"id": i, "name": n} for i, n in taxonomy.items()],
        "images": images_out,
        "annotations": anns_out,
    }
    report = {
        "dataset": dataset,
        "source_images": len(src["images"]),
        "kept_images": len(images_out),
        "kept_annotations": len(anns_out),
        "rejected_images": dict(rejected_images),
        "rejected_annotations": dict(rejected_anns),
        "rescaled_images": n_rescaled,
        "bbox_mismatch_recomputed": n_bbox_mismatch,
        "images_with_unverified_objects": sum(i["has_unverified_objects"] for i in images_out),
        "kept_instances_per_class": {n: kept_per_class[n] for n in taxonomy.values()},
        "source_label_actions": label_stats,
        "mapping_entries_not_in_source": unused_mapping,
    }
    return coco, report
