"""Build per-source image list files over the exported YOLO dataset.

Ultralytics accepts a .txt file of image paths as a split. Paths starting with "./" are resolved relative to the
list file's folder, so the lists stay valid when datasets/yolo is copied to another machine. Labels are found by
Ultralytics through the /images/ -> /labels/ substitution, so no data is duplicated.

Source is read from the exported file name prefix (`<source>_...`), set by export_yolo.
"""
from __future__ import annotations

from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def source_of_file(name: str, sources: list[str]) -> str | None:
    for s in sources:
        if name.startswith(f"{s}_"):
            return s
    return None


def build_lists(yolo_dir: Path, sources: list[str]) -> dict[str, int]:
    """Write {split}_{source}.txt and {split}_all.txt into yolo_dir. Returns image counts per list."""
    yolo_dir = Path(yolo_dir)
    counts: dict[str, int] = {}
    for split in SPLITS:
        folder = yolo_dir / "images" / split
        if not folder.is_dir():
            raise FileNotFoundError(f"missing {folder}; run scripts/export_yolo.py first")
        files = sorted(p.name for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not files:
            raise ValueError(f"no images in {folder}")
        per_source: dict[str, list[str]] = {s: [] for s in sources}
        for name in files:
            src = source_of_file(name, sources)
            if src is None:
                raise ValueError(f"{folder / name}: file name has no known source prefix {sources}")
            per_source[src].append(name)
        groups = {"all": files, **per_source}
        for key, names in groups.items():
            if not names:
                continue
            lines = [f"./images/{split}/{n}" for n in names]
            (yolo_dir / f"{split}_{key}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            counts[f"{split}_{key}"] = len(names)
    return counts
