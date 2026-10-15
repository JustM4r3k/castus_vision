"""Merge several internal COCO files (one per source dataset) into one."""
from __future__ import annotations

from pathlib import Path

from castus_vision.coco_io import load_coco, sha256_file


def merge_cocos(paths: list[Path]) -> dict:
    """Merge internal COCO files. Each image gets a unique `key` = "<source>/<file_name>"."""
    categories = None
    roots: dict[str, str] = {}
    images, annotations, inputs = [], [], []
    seen_keys: set[str] = set()

    for path in paths:
        coco = load_coco(path)
        if categories is None:
            categories = coco["categories"]
        elif categories != coco["categories"]:
            raise ValueError(f"{path}: categories differ from the other inputs")
        for source, root in coco["info"]["image_roots"].items():
            if roots.setdefault(source, root) != root:
                raise ValueError(f"Source '{source}' has conflicting image roots")
        inputs.append({"file": str(path), "sha256": sha256_file(path),
                       "dataset": coco["info"]["dataset"],
                       "source_json_sha256": coco["info"].get("source_json_sha256"),
                       "mapping_sha256": coco["info"].get("mapping_sha256")})

        id_map = {}
        for im in coco["images"]:
            key = f"{im['source']}/{im['file_name']}"
            if key in seen_keys:
                raise ValueError(f"Duplicate image key {key}")
            seen_keys.add(key)
            new = dict(im, id=len(images) + 1, key=key)
            id_map[im["id"]] = new["id"]
            images.append(new)
        for a in coco["annotations"]:
            annotations.append(dict(a, id=len(annotations) + 1, image_id=id_map[a["image_id"]]))

    return {"info": {"image_roots": roots, "inputs": inputs},
            "categories": categories, "images": images, "annotations": annotations}
