"""Group-aware, seeded, class-balanced train/val/test assignment with frozen-split support."""
from __future__ import annotations

import random
from collections import Counter

import numpy as np

SPLITS = ("train", "val", "test")
PRIORITY = {"test": 2, "val": 1, "train": 0}  # when a group spans splits, the stricter one wins


def assign_splits(coco: dict, groups: dict, ratios: dict, seed: int,
                  existing: dict | None = None) -> tuple[dict, dict, list]:
    """Return (assignment {key: split}, excluded {key: reason}, conflicts).

    - Only images with at least one annotation are eligible. Unlabeled images are never
      assumed to be negatives; they are reported as excluded.
    - All images of a duplicate group get the same split.
    - `existing` ({key: split}) pins previously assigned images, so a frozen test set
      is never reshuffled when data is added.
    """
    if abs(sum(ratios.values()) - 1.0) > 1e-6 or set(ratios) != set(SPLITS):
        raise ValueError(f"ratios must cover {SPLITS} and sum to 1, got {ratios}")
    existing = existing or {}
    cat_ids = [c["id"] for c in coco["categories"]]
    sources = sorted({im["source"] for im in coco["images"]})
    dims = ["images"] + [f"class:{c}" for c in cat_ids] + [f"source:{s}" for s in sources]
    dim_idx = {d: i for i, d in enumerate(dims)}

    vec = {}
    excluded = {}
    n_ann = Counter(a["image_id"] for a in coco["annotations"])
    cls_count: dict[int, Counter] = {}
    for a in coco["annotations"]:
        cls_count.setdefault(a["image_id"], Counter())[a["category_id"]] += 1
    for im in coco["images"]:
        if n_ann[im["id"]] == 0:
            excluded[im["key"]] = "no_annotations"
            continue
        v = np.zeros(len(dims))
        v[0] = 1
        for c, n in cls_count[im["id"]].items():
            v[dim_idx[f"class:{c}"]] = n
        v[dim_idx[f"source:{im['source']}"]] = 1
        vec[im["key"]] = v

    members: dict[str, list[str]] = {}
    for key in vec:
        members.setdefault(groups[key], []).append(key)
    gvec = {g: sum(vec[k] for k in ks) for g, ks in members.items()}

    total = sum(gvec.values())
    target = {s: ratios[s] * total for s in SPLITS}
    cur = {s: np.zeros(len(dims)) for s in SPLITS}
    assignment, conflicts, pending = {}, [], []

    for g in sorted(members):
        known = {existing[k] for k in members[g] if k in existing}
        if not known:
            pending.append(g)
            continue
        split = max(known, key=PRIORITY.get)
        if len(known) > 1:
            conflicts.append({"group": g, "existing_splits": sorted(known), "assigned": split})
        for k in members[g]:
            assignment[k] = split
        cur[split] += gvec[g]

    rng = random.Random(seed)
    rng.shuffle(pending)
    safe_total = np.where(total > 0, total, 1)
    pending.sort(key=lambda g: -float((gvec[g] / safe_total).sum()))  # stable: rare/large groups first
    for g in pending:
        best, best_cost = None, None
        for s in SPLITS:
            if ratios[s] == 0:
                continue
            tgt = np.maximum(target[s], 1.0)
            cost = float(((((cur[s] + gvec[g]) / tgt - 1) ** 2 - (cur[s] / tgt - 1) ** 2))[total > 0].sum())
            if best_cost is None or cost < best_cost:
                best, best_cost = s, cost
        for k in members[g]:
            assignment[k] = best
        cur[best] += gvec[g]
    return assignment, excluded, conflicts


def check_leakage(assignment: dict, groups: dict, hashes: dict) -> dict:
    """Count duplicate groups and identical files that span more than one split."""
    by_group: dict[str, set] = {}
    by_sha: dict[str, set] = {}
    for k, s in assignment.items():
        by_group.setdefault(groups[k], set()).add(s)
        by_sha.setdefault(hashes[k]["sha256"], set()).add(s)
    return {"groups_spanning_splits": sum(len(v) > 1 for v in by_group.values()),
            "identical_files_spanning_splits": sum(len(v) > 1 for v in by_sha.values())}
