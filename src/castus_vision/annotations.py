"""Geometry validation and normalisation of single COCO instance annotations."""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils

BBOX_TOLERANCE_PX = 2.0
COORD_TOLERANCE_PX = 2.0


class AnnotationError(ValueError):
    """Raised with a short machine-readable reason when an annotation is unusable."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def normalize_annotation(segmentation, bbox, width: int, height: int,
                         sx: float = 1.0, sy: float = 1.0) -> dict:
    """Validate a segmentation and return {segmentation, bbox, area, bbox_mismatch}.

    Polygon coordinates are multiplied by (sx, sy) (used when the image on disk is a
    uniformly rescaled version of the annotated image) and clipped to the image.
    RLE masks must already match (height, width) and cannot be rescaled.
    The returned bbox/area are always derived from the mask, not copied from the source.
    """
    if isinstance(segmentation, list):
        if not segmentation:
            raise AnnotationError("empty_segmentation")
        polygons = []
        for poly in segmentation:
            if not isinstance(poly, list) or len(poly) < 6 or len(poly) % 2:
                raise AnnotationError("bad_polygon")
            arr = np.asarray(poly, dtype=float)
            if not np.isfinite(arr).all():
                raise AnnotationError("non_finite_coordinates")
            arr[0::2] *= sx
            arr[1::2] *= sy
            if (arr[0::2].min() < -COORD_TOLERANCE_PX or arr[0::2].max() > width + COORD_TOLERANCE_PX
                    or arr[1::2].min() < -COORD_TOLERANCE_PX or arr[1::2].max() > height + COORD_TOLERANCE_PX):
                raise AnnotationError("out_of_bounds")
            arr[0::2] = arr[0::2].clip(0, width)
            arr[1::2] = arr[1::2].clip(0, height)
            polygons.append(arr.tolist())
        rle = mask_utils.merge(mask_utils.frPyObjects(polygons, height, width))
        out_seg = polygons
    elif isinstance(segmentation, dict):
        if sx != 1.0 or sy != 1.0:
            raise AnnotationError("rle_cannot_be_rescaled")
        if list(segmentation.get("size", [])) != [height, width]:
            raise AnnotationError("rle_size_mismatch")
        counts = segmentation.get("counts")
        if isinstance(counts, list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        elif isinstance(counts, (str, bytes)):
            rle = {"size": [height, width],
                   "counts": counts.encode() if isinstance(counts, str) else counts}
        else:
            raise AnnotationError("bad_rle")
        out_seg = {"size": [height, width],
                   "counts": rle["counts"].decode() if isinstance(rle["counts"], bytes) else rle["counts"]}
    else:
        raise AnnotationError("unsupported_segmentation")

    area = float(mask_utils.area(rle))
    if area <= 0:
        raise AnnotationError("empty_mask")
    x, y, w, h = (float(v) for v in mask_utils.toBbox(rle))

    mismatch = False
    if bbox is not None and len(bbox) == 4:
        sb = [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]
        mismatch = any(abs(a - b) > BBOX_TOLERANCE_PX for a, b in zip(sb, (x, y, w, h)))
    return {"segmentation": out_seg, "bbox": [x, y, w, h], "area": area, "bbox_mismatch": mismatch}
