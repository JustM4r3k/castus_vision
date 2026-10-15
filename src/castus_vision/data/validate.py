"""Validate an internal-format COCO file against the taxonomy and the images on disk."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from castus_vision.annotations import AnnotationError, normalize_annotation
from castus_vision.coco_io import image_path, load_coco, upright_size
from castus_vision.taxonomy import DEFAULT_TAXONOMY, load_taxonomy

MAX_EXAMPLES = 20


def validate_coco(coco_path: Path, taxonomy_path: Path = DEFAULT_TAXONOMY,
                  check_images: bool = True) -> dict:
    """Return {'errors': {reason: count}, 'examples': {...}, 'ok': bool}."""
    coco = load_coco(coco_path)
    taxonomy = load_taxonomy(taxonomy_path)
    errors: Counter = Counter()
    examples: dict[str, list] = {}

    def err(reason: str, detail) -> None:
        errors[reason] += 1
        examples.setdefault(reason, [])
        if len(examples[reason]) < MAX_EXAMPLES:
            examples[reason].append(detail)

    cats = {c["id"]: c["name"] for c in coco["categories"]}
    if cats != taxonomy:
        err("categories_differ_from_taxonomy", {"file": cats, "taxonomy": taxonomy})

    images = {}
    for im in coco["images"]:
        if im["id"] in images:
            err("duplicate_image_id", im["id"])
        images[im["id"]] = im
        if check_images:
            p = image_path(coco, im)
            if not p.is_file():
                err("missing_image_file", str(p))
            else:
                try:
                    if upright_size(p) != (im["width"], im["height"]):
                        err("image_size_mismatch", str(p))
                except Exception:
                    err("unreadable_image", str(p))

    seen_ann = set()
    for a in coco["annotations"]:
        if a["id"] in seen_ann:
            err("duplicate_annotation_id", a["id"])
        seen_ann.add(a["id"])
        if a["category_id"] not in taxonomy:
            err("unknown_category_id", a["id"])
        im = images.get(a["image_id"])
        if im is None:
            err("annotation_without_image", a["id"])
            continue
        try:
            normalize_annotation(a["segmentation"], a.get("bbox"), im["width"], im["height"])
        except AnnotationError as e:
            err(f"invalid_annotation:{e.reason}", a["id"])

    return {"images": len(images), "annotations": len(coco["annotations"]),
            "errors": dict(errors), "examples": examples, "ok": not errors}
