"""Export the merged dataset to YOLO segmentation format.

    python scripts/export_yolo.py
Writes datasets/yolo/{images,labels}/{train,val,test} and datasets/yolo/data.yaml.
"""
import argparse
import json
from pathlib import Path

from castus_vision.coco_io import load_coco, save_json
from castus_vision.data.export_yolo import export_yolo


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--merged", type=Path, default=Path("datasets/merged/merged.coco.json"))
    p.add_argument("--splits", type=Path, default=Path("datasets/merged/splits.json"))
    p.add_argument("--out-dir", type=Path, default=Path("datasets/yolo"))
    p.add_argument("--link", action="store_true", help="hard-link unmodified images instead of copying")
    a = p.parse_args()

    coco = load_coco(a.merged)
    manifest = json.loads(a.splits.read_text())
    assignment = {k: v["split"] for k, v in manifest["images"].items()}
    stats = export_yolo(coco, assignment, a.out_dir, a.link)
    save_json({"dataset_version": manifest["dataset_version"], "export_stats": stats}, a.out_dir / "dataset_info.json")
    print(json.dumps(stats, indent=2))
    print(f"dataset_version={manifest['dataset_version']}  data.yaml: {a.out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
