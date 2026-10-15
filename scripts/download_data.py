"""Download the raw TACO and BePLi v2 datasets (for the GPU VM).

    python scripts/download_data.py --raw-root /data/castus_raw
    python scripts/download_data.py --raw-root /data/castus_raw --datasets taco
Re-running is safe: finished files are skipped, the BePLi archive download is resumed, and TACO images that only
exist as 640 px fallbacks are retried for the original.

Layout created (what scripts/import_dataset.py --raw-root expects):
    <raw-root>/TACO/data/annotations.json + batch_*/*.jpg
    <raw-root>/BePLi_dataset_v2/plastic_coco/{annotation,images/original_images,README}

Sources and terms:
  * TACO annotations: https://github.com/pedropro/TACO (images are fetched from the Flickr URLs listed in it;
    individual images carry their own Flickr licences).
  * BePLi v2: https://www.seanoe.org/data/00858/96963/ (JAMSTEC). Its terms limit use to non-profit,
    non-commercial research. Check this before using models trained on it in a product.

Note: Flickr answers 429 to the plain urllib client but not to `requests`, so `requests` is used (as in the
upstream TACO download script).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, UnidentifiedImageError

TACO_ANN_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
# SHA-256 of the annotations.json the frozen splits were built from.
TACO_ANN_SHA256 = "ac1ec605c9e1fcda767970de3457c69b4a43b62c02aca8c4c80f51d1b5ae8449"
BEPLI_URL = "https://www.seanoe.org/data/00858/96963/data/106157.tar.gz"
BEPLI_ARCHIVE_BYTES = 9489309612
BEPLI_EXPECTED_IMAGES = 3722
BEPLI_PREFIX = "BePLi_dataset_v2/plastic_coco/"

_lock = threading.Lock()


class Permanent(Exception):
    """The URL will not work (404/410/not an image); retrying is pointless."""


_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()  # keep-alive: avoids a new TLS handshake per image
    return _local.s


def _get(url: str, max_attempts: int = 12) -> bytes:
    # Flickr randomly answers 429 for a fraction of requests; a short jittered retry succeeds.
    for attempt in range(max_attempts):
        try:
            r = _session().get(url, timeout=30)
        except requests.RequestException:
            time.sleep(min(2 ** attempt, 15))
            continue
        if r.status_code == 200:
            return r.content
        if r.status_code in (404, 410, 403):
            raise Permanent(f"HTTP {r.status_code}")
        time.sleep(random.uniform(0.3, 1.0) * min(1 + attempt, 6))
    raise RuntimeError(f"gave up on {url} (rate limited or unreachable)")


def _save_image(data: bytes, dest: Path) -> None:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise Permanent(f"not a valid image: {e}") from e
    tmp = dest.with_suffix(dest.suffix + ".part")
    exif = img.info.get("exif")
    img.save(tmp, format="JPEG", exif=exif) if exif else img.save(tmp, format="JPEG")
    tmp.replace(dest)


def _fetch_taco_image(im: dict, dest: Path) -> str:
    """Original first; the 640 px copy only if the original is permanently gone."""
    try:
        _save_image(_get(im["flickr_url"]), dest)
        return "original"
    except Permanent:
        _save_image(_get(im["flickr_640_url"]), dest)
        return "640px"


def download_taco(raw_root: Path, workers: int) -> dict:
    data_dir = raw_root / "TACO" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    ann_path = data_dir / "annotations.json"
    if not ann_path.is_file():
        print(f"TACO: downloading annotations from {TACO_ANN_URL}")
        ann_path.write_bytes(_get(TACO_ANN_URL))
    sha = hashlib.sha256(ann_path.read_bytes()).hexdigest()
    if sha != TACO_ANN_SHA256:
        print(f"WARNING: TACO annotations.json differs from the version the frozen splits were built for\n"
              f"  expected {TACO_ANN_SHA256}\n  got      {sha}", file=sys.stderr)

    report_path = raw_root / "TACO" / "download_report.json"
    prev = json.loads(report_path.read_text()) if report_path.is_file() else {}
    fallbacks = set(prev.get("fallback_640px", []))  # retried so a temporary glitch does not stick
    images = json.loads(ann_path.read_text(encoding="utf-8"))["images"]
    todo = [im for im in images if not (data_dir / im["file_name"]).is_file() or im["file_name"] in fallbacks]
    print(f"TACO: {len(images)} images, {len(images) - len(todo)} already present, {len(todo)} to download")

    failed, new_fallbacks, done = [], set(), 0

    def work(im):
        dest = data_dir / im["file_name"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        return _fetch_taco_image(im, dest)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(work, im): im for im in todo}
        for f in as_completed(futures):
            name = futures[f]["file_name"]
            try:
                if f.result() == "640px":
                    new_fallbacks.add(name)
            except (Permanent, RuntimeError) as e:
                failed.append({"file_name": name, "error": str(e)[:300]})
            done += 1
            if done % 100 == 0 or done == len(todo):
                print(f"  TACO {done}/{len(todo)}  (failed {len(failed)}, 640px fallbacks {len(new_fallbacks)})", flush=True)

    kept = {n for n in fallbacks if n not in {im["file_name"] for im in todo}}
    summary = {"images": len(images), "failed": failed, "fallback_640px": sorted(new_fallbacks | kept)}
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failed:
        print(f"WARNING: {len(failed)} TACO images failed (see {report_path}); re-run to retry. "
              f"The importer skips missing files.", file=sys.stderr)
    if summary["fallback_640px"]:
        print(f"note: {len(summary['fallback_640px'])} images only exist as 640 px copies on Flickr; the importer "
              f"rescales their annotations (recorded as rescaled_from).")
    return summary


def _download_resumable(url: str, dest: Path, expected: int) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    while (part.stat().st_size if part.is_file() else 0) < expected:
        have = part.stat().st_size if part.is_file() else 0
        try:
            with requests.get(url, headers={"Range": f"bytes={have}-"}, stream=True, timeout=120) as r:
                r.raise_for_status()
                if have and r.status_code != 206:
                    have = 0  # server ignored Range: start over
                last = time.time()
                with open(part, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        if time.time() - last > 10:
                            print(f"  BePLi archive {f.tell() / 1e9:.2f}/{expected / 1e9:.2f} GB", flush=True)
                            last = time.time()
        except requests.RequestException as e:
            print(f"  connection problem ({e}); retrying in 5 s", file=sys.stderr)
            time.sleep(5)
    part.replace(dest)


def _wanted(member: tarfile.TarInfo) -> bool:
    n = member.name.lstrip("./")
    return (n.startswith(BEPLI_PREFIX + "images/original_images/") or n.startswith(BEPLI_PREFIX + "annotation/")
            or n == BEPLI_PREFIX + "README")


def download_bepli(raw_root: Path, keep_archive: bool) -> dict:
    target = raw_root / "BePLi_dataset_v2"
    img_dir = target / "plastic_coco" / "images" / "original_images"
    ann = target / "plastic_coco" / "annotation" / "all_plastic_coco.json"
    if ann.is_file() and img_dir.is_dir() and sum(1 for _ in img_dir.iterdir()) >= BEPLI_EXPECTED_IMAGES:
        print("BePLi: already present, skipping")
        return {"skipped": True}
    raw_root.mkdir(parents=True, exist_ok=True)
    archive = raw_root / "bepli_v2.tar.gz"
    if not archive.is_file():
        print(f"BePLi: downloading ~{BEPLI_ARCHIVE_BYTES / 1e9:.1f} GB from SEANOE (resumable)")
        _download_resumable(BEPLI_URL, archive, BEPLI_ARCHIVE_BYTES)
    print("BePLi: extracting annotations and original_images (skipping the duplicate train/val/test image copies)")
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.isfile() and _wanted(member):
                member.name = member.name.lstrip("./")
                tar.extract(member, raw_root, filter="data")
    n = sum(1 for _ in img_dir.iterdir())
    if n != BEPLI_EXPECTED_IMAGES or not ann.is_file():
        raise RuntimeError(f"BePLi extraction incomplete: {n}/{BEPLI_EXPECTED_IMAGES} images, annotation={ann.is_file()}")
    if not keep_archive:
        archive.unlink()
    print(f"BePLi: {n} images ready")
    return {"images": n}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-root", type=Path, default=Path("datasets/raw"))
    p.add_argument("--datasets", nargs="+", choices=["taco", "bepli"], default=["taco", "bepli"])
    p.add_argument("--workers", type=int, default=1,
                   help="parallel TACO downloads; 1 (sequential, like the upstream script) was fastest because "
                        "Flickr throttles parallel connections")
    p.add_argument("--keep-archive", action="store_true", help="keep the 9.5 GB BePLi tar.gz after extraction")
    a = p.parse_args()
    if "taco" in a.datasets:
        download_taco(a.raw_root, a.workers)
    if "bepli" in a.datasets:
        download_bepli(a.raw_root, a.keep_archive)


if __name__ == "__main__":
    main()
