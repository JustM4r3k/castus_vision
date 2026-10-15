"""Create train_taco.txt, train_all.txt, val_bepli.txt ... inside the exported YOLO dataset.

    python scripts/make_yolo_lists.py --yolo-dir datasets/yolo --sources taco bepli
"""
import argparse
import json
from pathlib import Path

from castus_vision.data.lists import build_lists


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yolo-dir", type=Path, default=Path("datasets/yolo"))
    p.add_argument("--sources", nargs="+", default=["taco", "bepli"])
    a = p.parse_args()
    print(json.dumps(build_lists(a.yolo_dir, a.sources), indent=2))


if __name__ == "__main__":
    main()
