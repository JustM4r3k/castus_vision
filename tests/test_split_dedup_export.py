import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from castus_vision.data.dedup import cluster_images
from castus_vision.data.export_yolo import annotation_to_polygon
from castus_vision.data.split import assign_splits, check_leakage


def _coco(n_groups=60, per_group=3):
    images, anns = [], []
    for g in range(n_groups):
        for j in range(per_group):
            i = len(images) + 1
            images.append({"id": i, "key": f"s{g % 2}/{g}_{j}.jpg", "source": f"s{g % 2}", "width": 100, "height": 100})
            anns.append({"id": i, "image_id": i, "category_id": g % 3, "area": 100.0})
    cats = [{"id": c, "name": f"c{c}"} for c in range(3)]
    return {"images": images, "annotations": anns, "categories": cats}


def test_split_keeps_groups_together_and_is_deterministic():
    coco = _coco()
    groups = {im["key"]: im["key"].split("_")[0] for im in coco["images"]}
    hashes = {k: {"sha256": k} for k in groups}
    r = {"train": 0.8, "val": 0.1, "test": 0.1}
    a1, ex, conf = assign_splits(coco, groups, r, seed=1)
    a2, _, _ = assign_splits(coco, groups, r, seed=1)
    assert a1 == a2 and not ex and not conf
    assert check_leakage(a1, groups, hashes) == {"groups_spanning_splits": 0, "identical_files_spanning_splits": 0}
    assert set(a1.values()) == {"train", "val", "test"}


def test_frozen_assignment_is_respected_and_conflict_reported():
    coco = _coco(6, 2)
    groups = {im["key"]: im["key"].split("_")[0] for im in coco["images"]}
    k0, k1 = "s0/0_0.jpg", "s0/0_1.jpg"
    existing = {k0: "train", k1: "test"}
    a, _, conf = assign_splits(coco, groups, {"train": 0.8, "val": 0.1, "test": 0.1}, 1, existing)
    assert a[k0] == a[k1] == "test" and len(conf) == 1


def test_unlabeled_images_excluded():
    coco = _coco(2, 1)
    coco["annotations"].pop()
    groups = {im["key"]: im["key"] for im in coco["images"]}
    a, ex, _ = assign_splits(coco, groups, {"train": 1.0, "val": 0.0, "test": 0.0}, 1)
    assert list(ex.values()) == ["no_annotations"] and len(a) == 1


def test_cluster_by_hash():
    base = "00" * 32
    near = "01" + "00" * 31  # 1 bit apart
    far = "ff" * 32
    h = {"a": {"sha256": "1", "dhash": base}, "b": {"sha256": "2", "dhash": near},
         "c": {"sha256": "3", "dhash": far}, "d": {"sha256": "1", "dhash": far}}
    groups, rep = cluster_images(h, threshold=4)
    assert groups["a"] == groups["b"] == groups["d"] == groups["c"]  # d links a (same file) and c (same hash)
    h["d"] = {"sha256": "4", "dhash": far}
    groups, _ = cluster_images(h, threshold=4)
    assert groups["a"] == groups["b"] != groups["c"] == groups["d"]


def test_polygon_conversion_rle_and_multipart():
    m = np.zeros((40, 40), np.uint8)
    m[5:15, 5:15] = 1
    m[30:32, 30:32] = 1
    rle = mask_utils.encode(np.asfortranarray(m))
    poly, note = annotation_to_polygon({"segmentation": {"size": [40, 40], "counts": rle["counts"].decode()}}, 40, 40)
    assert len(poly) >= 8 and note == ""  # small second part is under 10% of area
    single = np.zeros((40, 40), np.uint8)
    single[7, 9] = 1
    rle = mask_utils.encode(np.asfortranarray(single))
    poly, note = annotation_to_polygon({"segmentation": {"size": [40, 40], "counts": rle["counts"].decode()}}, 40, 40)
    assert note == "tiny_pixel_footprint" and len(poly) == 8
