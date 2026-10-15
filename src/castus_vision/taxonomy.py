"""Fixed class taxonomy and dataset-specific label mappings."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAXONOMY = REPO_ROOT / "configs" / "taxonomy.yaml"
MAPPINGS_DIR = REPO_ROOT / "configs" / "mappings"

IGNORE = "ignore"
REVIEW = "review"


def load_taxonomy(path: Path | str = DEFAULT_TAXONOMY) -> dict[int, str]:
    """Return {class_id: class_name}. IDs must be contiguous from 0."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    classes = {int(k): str(v) for k, v in data["classes"].items()}
    if sorted(classes) != list(range(len(classes))):
        raise ValueError(f"Class IDs must be contiguous from 0, got {sorted(classes)}")
    if len(set(classes.values())) != len(classes):
        raise ValueError("Duplicate class names in taxonomy")
    return classes


def load_mapping(path: Path | str, taxonomy: dict[int, str]) -> dict[str, str]:
    """Return {source_label: castus_class_name | 'ignore' | 'review'}."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    valid = set(taxonomy.values()) | {IGNORE, REVIEW}
    mapping = {str(k): str(v) for k, v in data["labels"].items()}
    bad = {k: v for k, v in mapping.items() if v not in valid}
    if bad:
        raise ValueError(f"Mapping {path} has invalid targets: {bad}")
    return mapping
