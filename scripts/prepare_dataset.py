"""One command to build the training dataset on a fresh machine (e.g. the GPU VM).

    python scripts/prepare_dataset.py --raw-root /data/castus_raw
    python scripts/prepare_dataset.py --raw-root /data/castus_raw --skip-download

Steps: download TACO + BePLi -> import with the class mappings -> validate -> install the frozen split manifest
(configs/splits/splits.json) -> merge/dedup/split -> verify the splits equal the frozen ones -> YOLO export -> image lists.
Output: datasets/yolo (train with configs/train/*.yaml).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(*args: str) -> None:
    cmd = [PY, *args]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-root", type=Path, default=Path("datasets/raw"))
    p.add_argument("--frozen-splits", type=Path, default=Path("configs/splits/splits.json"))
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--workers", type=int, default=1, help="parallel TACO downloads (1 is fastest; Flickr throttles parallel use)")
    a = p.parse_args()
    raw = a.raw_root.resolve()

    if not a.skip_download:
        run("scripts/download_data.py", "--raw-root", str(raw), "--workers", str(a.workers))

    for ds in ("taco", "bepli"):
        run("scripts/import_dataset.py", "--dataset", ds, "--raw-root", str(raw))
        run("scripts/validate_dataset.py", f"datasets/internal/{ds}.coco.json")

    frozen_src = ROOT / a.frozen_splits
    if not frozen_src.is_file():
        raise SystemExit(f"frozen split manifest {frozen_src} not found")
    merged = ROOT / "datasets/merged"
    merged.mkdir(parents=True, exist_ok=True)
    shutil.copy2(frozen_src, merged / "splits.json")  # build_splits keeps assignments found in this file
    run("scripts/build_splits.py")

    frozen = json.loads(frozen_src.read_text())
    built = json.loads((merged / "splits.json").read_text())
    changed = [k for k, v in built["images"].items() if k in frozen["images"] and frozen["images"][k]["split"] != v["split"]]
    new = [k for k in built["images"] if k not in frozen["images"]]
    missing = [k for k in frozen["images"] if k not in built["images"]]
    print(f"\nsplit check: changed={len(changed)} new_images={len(new)} missing_vs_frozen={len(missing)}")
    if changed:
        raise SystemExit("splits differ from the frozen manifest; refusing to continue")
    if new or missing:
        print("WARNING: image set differs from the frozen manifest (unavailable Flickr images or new data). "
              "Results are not strictly comparable to runs made on the original dataset.", file=sys.stderr)
    same = built["dataset_version"] == frozen["dataset_version"]
    print(f"dataset_version: frozen={frozen['dataset_version']} built={built['dataset_version']} "
          f"-> {'IDENTICAL' if same else 'DIFFERENT'}")

    run("scripts/export_yolo.py")
    run("scripts/make_yolo_lists.py")
    print("\nDone. Train with: python scripts/train_yolo.py --config configs/train/A_taco_only.yaml --set train.device=0")


if __name__ == "__main__":
    main()
