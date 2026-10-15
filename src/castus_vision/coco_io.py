"""Reading/writing the internal COCO files and resolving image paths."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


def load_coco(path: Path | str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=False)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_path(coco: dict, image: dict) -> Path:
    """Absolute path of an image in an internal COCO file (root stored per source)."""
    return Path(coco["info"]["image_roots"][image["source"]]) / image["file_name"]


def upright_size(path: Path | str) -> tuple[int, int]:
    """(width, height) as displayed after applying EXIF orientation, reading only the header."""
    with Image.open(path) as im:
        w, h = im.size
        if im.getexif().get(274, 1) in (5, 6, 7, 8):
            w, h = h, w
    return w, h
