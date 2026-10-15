"""Validate an internal COCO file. Exit code 1 if any error is found.

    python scripts/validate_dataset.py datasets/internal/taco.coco.json
"""
import argparse
import json
import sys
from pathlib import Path

from castus_vision.data.validate import validate_coco


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("coco", type=Path)
    p.add_argument("--no-check-images", action="store_true")
    a = p.parse_args()
    result = validate_coco(a.coco, check_images=not a.no_check_images)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
