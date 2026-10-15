"""Import TACO or BePLi (COCO) into the internal format.

Examples (run from castus_vision/):
    python scripts/import_dataset.py --dataset taco
    python scripts/import_dataset.py --dataset bepli
"""
import argparse
import json
from pathlib import Path

from castus_vision.coco_io import save_json
from castus_vision.data.import_coco import import_coco
from castus_vision.taxonomy import MAPPINGS_DIR

WORKSPACE = Path(__file__).resolve().parents[2]


def presets(raw_root: Path) -> dict:
    return {
        "taco": (raw_root / "TACO/data/annotations.json", raw_root / "TACO/data"),
        "bepli": (raw_root / "BePLi_dataset_v2/plastic_coco/annotation/all_plastic_coco.json",
                  raw_root / "BePLi_dataset_v2/plastic_coco/images/original_images"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="dataset name; taco and bepli have default paths")
    p.add_argument("--raw-root", type=Path, default=WORKSPACE,
                   help="folder holding TACO/ and BePLi_dataset_v2/ (same layout scripts/download_data.py creates)")
    p.add_argument("--source-json", type=Path)
    p.add_argument("--image-root", type=Path)
    p.add_argument("--mapping", type=Path, help="default: configs/mappings/<dataset>.yaml")
    p.add_argument("--out-dir", type=Path, default=Path("datasets/internal"))
    p.add_argument("--no-check-images", action="store_true", help="skip image existence/size checks")
    a = p.parse_args()

    preset = presets(a.raw_root).get(a.dataset, (None, None))
    source_json = a.source_json or preset[0]
    image_root = a.image_root or preset[1]
    if source_json is None or image_root is None:
        p.error("--source-json and --image-root are required for custom datasets")
    mapping = a.mapping or MAPPINGS_DIR / f"{a.dataset}.yaml"

    coco, report = import_coco(a.dataset, source_json, image_root, mapping,
                               check_images=not a.no_check_images)
    save_json(coco, a.out_dir / f"{a.dataset}.coco.json")
    save_json(report, a.out_dir / f"{a.dataset}.import_report.json")
    summary = {k: v for k, v in report.items() if k not in ("source_label_actions",)}
    print(json.dumps(summary, indent=2))
    print(f"Wrote {a.out_dir / (a.dataset + '.coco.json')}")


if __name__ == "__main__":
    main()
