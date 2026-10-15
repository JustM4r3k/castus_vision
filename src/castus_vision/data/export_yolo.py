"""Export the merged COCO dataset + split manifest to the Ultralytics YOLO segmentation layout.

Images with a non-trivial EXIF orientation are re-saved upright so labels and pixels agree
regardless of the training library. Other images are copied (or hard-linked).
"""
from __future__ import annotations

import os
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageOps
from pycocotools import mask as mask_utils

from castus_vision.coco_io import image_path
from castus_vision.data.split import SPLITS


def annotation_to_polygon(ann: dict, width: int, height: int) -> tuple[list[float] | None, str]:
    """Return (flat normalised polygon [x1,y1,...] or None, note).

    Notes: "" normal, "multi_part_largest_kept" when parts were dropped,
    "tiny_pixel_footprint" when the mask is too small for a contour polygon.
    """
    seg = ann["segmentation"]
    if isinstance(seg, list) and len(seg) == 1:
        pts = np.asarray(seg[0], dtype=np.float32).reshape(-1, 2)
        return _normalise(pts, width, height), ""

    if isinstance(seg, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(seg, height, width))
    else:
        counts = seg["counts"]
        rle = {"size": seg["size"], "counts": counts.encode() if isinstance(counts, str) else counts}
    mask = np.ascontiguousarray(mask_utils.decode(rle).astype(np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "empty_mask"
    note = ""
    if len(contours) > 1:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        kept = float(mask[cv2.drawContours(np.zeros_like(mask), contours[:1], -1, 1, -1) > 0].sum())
        if kept / max(float(mask.sum()), 1.0) < 0.9:
            note = "multi_part_largest_kept"
    c = contours[0]
    if len(c) < 3:
        x, y, w, h = cv2.boundingRect(c)
        pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        return _normalise(pts, width, height), "tiny_pixel_footprint"
    c = cv2.approxPolyDP(c, 0.5, True)
    if len(c) < 3:
        x, y, w, h = cv2.boundingRect(c)
        pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        return _normalise(pts, width, height), "tiny_pixel_footprint"
    return _normalise(c.reshape(-1, 2).astype(np.float32), width, height), note


def _normalise(pts: np.ndarray, width: int, height: int) -> list[float]:
    pts = pts.copy()
    pts[:, 0] = (pts[:, 0] / width).clip(0, 1)
    pts[:, 1] = (pts[:, 1] / height).clip(0, 1)
    return pts.flatten().tolist()


def _write_image(src: Path, dst: Path, link: bool) -> None:
    with Image.open(src) as im:
        needs_transpose = im.getexif().get(274, 1) in range(2, 9)
        if needs_transpose:
            fmt = im.format
            up = ImageOps.exif_transpose(im)
            up.save(dst, format=fmt, **({"quality": 95} if fmt == "JPEG" else {}))
            return
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def export_yolo(coco: dict, assignment: dict, out_dir: Path, link: bool = False) -> dict:
    out_dir = Path(out_dir)
    for sub in ("images", "labels"):
        shutil.rmtree(out_dir / sub, ignore_errors=True)
    for s in SPLITS:
        (out_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

    anns: dict[int, list] = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)

    stats = Counter()
    for im in coco["images"]:
        split = assignment.get(im["key"])
        if not split:
            continue
        stem = im["key"].replace("/", "_")
        stem, ext = os.path.splitext(stem)
        _write_image(image_path(coco, im), out_dir / "images" / split / f"{stem}{ext.lower()}", link)
        lines = []
        for a in anns.get(im["id"], []):
            poly, note = annotation_to_polygon(a, im["width"], im["height"])
            if poly is None:
                stats["skipped_empty_mask"] += 1
                continue
            if note:
                stats[note] += 1
            lines.append(f"{a['category_id']} " + " ".join(f"{v:.6f}" for v in poly))
        (out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats[f"images_{split}"] += 1
        stats["instances"] += len(lines)

    names = {c["id"]: c["name"] for c in coco["categories"]}
    data = {"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val",
            "test": "images/test", "names": names}
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return dict(stats)
