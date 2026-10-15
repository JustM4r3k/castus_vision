import json

import pytest
from PIL import Image

from castus_vision.annotations import AnnotationError, normalize_annotation
from castus_vision.data.import_coco import import_coco
from castus_vision.data.validate import validate_coco
from castus_vision.taxonomy import load_taxonomy


def test_taxonomy_ids_fixed():
    t = load_taxonomy()
    assert t == {0: "can", 1: "plastic_bottle", 2: "glass", 3: "metal_other",
                 4: "paper_cardboard", 5: "plastic_other", 6: "other_trash"}


def test_polygon_ok_and_bbox_from_mask():
    r = normalize_annotation([[10, 10, 30, 10, 30, 20, 10, 20]], [0, 0, 5, 5], 100, 50)
    assert r["area"] == 200
    assert r["bbox"] == [10, 10, 20, 10]
    assert r["bbox_mismatch"]


def test_polygon_scaled():
    r = normalize_annotation([[100, 100, 300, 100, 300, 200, 100, 200]], None, 200, 100, 0.5, 0.5)
    assert r["bbox"] == [50, 50, 100, 50]


@pytest.mark.parametrize("seg,reason", [
    ([], "empty_segmentation"),
    ([[1, 2, 3, 4]], "bad_polygon"),
    ([[500, 500, 600, 500, 600, 600]], "out_of_bounds"),
    ([[5, 5, 10, 10, 15, 15]], "empty_mask"),  # collinear -> zero area
    ({"size": [9, 9], "counts": "x"}, "rle_size_mismatch"),
])
def test_invalid_annotations(seg, reason):
    with pytest.raises(AnnotationError) as e:
        normalize_annotation(seg, None, 100, 50)
    assert e.value.reason == reason


def _make_source(tmp_path):
    root = tmp_path / "imgs"
    root.mkdir()
    Image.new("RGB", (100, 50)).save(root / "a.jpg")
    Image.new("RGB", (50, 25)).save(root / "b.jpg")  # downscaled copy of a 100x50 annotation
    src = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 50},
                   {"id": 2, "file_name": "b.jpg", "width": 100, "height": 50},
                   {"id": 3, "file_name": "missing.jpg", "width": 10, "height": 10}],
        "categories": [{"id": 1, "name": "Can"}, {"id": 2, "name": "Mystery"}, {"id": 3, "name": "Unlabeled"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 10],
             "segmentation": [[10, 10, 30, 10, 30, 20, 10, 20]], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "segmentation": [[1, 1, 9, 1, 9, 9]], "iscrowd": 0},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [10, 10, 20, 10],
             "segmentation": [[10, 10, 30, 10, 30, 20, 10, 20]], "iscrowd": 0},
            {"id": 4, "image_id": 2, "category_id": 3, "segmentation": [[1, 1, 9, 1, 9, 9]], "iscrowd": 0},
        ],
    }
    (tmp_path / "src.json").write_text(json.dumps(src))
    (tmp_path / "map.yaml").write_text(
        "dataset: t\nlabels:\n  Can: can\n  Mystery: review\n  Unlabeled: ignore\n")
    return tmp_path / "src.json", root, tmp_path / "map.yaml"


def test_import_and_validate(tmp_path):
    src, root, mp = _make_source(tmp_path)
    coco, report = import_coco("t", src, root, mp)
    assert report["rejected_images"] == {"missing_file": 1}
    assert report["rescaled_images"] == 1
    assert [a["category_id"] for a in coco["annotations"]] == [0, 0]
    assert all(i["has_unverified_objects"] for i in coco["images"])  # review/ignore objects present
    assert coco["images"][1]["rescaled_from"] == [100, 50]
    assert coco["annotations"][1]["bbox"] == [5, 5, 10, 5]

    out = tmp_path / "t.coco.json"
    out.write_text(json.dumps(coco))
    assert validate_coco(out)["ok"]


def test_unmapped_label_is_error(tmp_path):
    src, root, mp = _make_source(tmp_path)
    mp.write_text("dataset: t\nlabels:\n  Can: can\n")
    with pytest.raises(ValueError, match="missing from mapping"):
        import_coco("t", src, root, mp)
