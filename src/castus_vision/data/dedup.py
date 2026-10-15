"""Exact and near-duplicate detection, producing leakage-safe image groups.

Near duplicates are found with a 256-bit difference hash (dHash) of the upright image.
Images whose Hamming distance is <= threshold (or whose file bytes are identical) are
linked, and connected components become groups. Splitting is done per group.
"""
from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from castus_vision.coco_io import image_path, sha256_file

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hash_one(args):
    key, path = args
    sha = sha256_file(path)
    with Image.open(path) as im:
        if im.format == "JPEG":
            im.draft("L", (128, 128))  # fast reduced-size decode
        gray = ImageOps.exif_transpose(im).convert("L").resize((17, 16), Image.Resampling.BOX)
    a = np.asarray(gray, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return key, sha, np.packbits(bits).tobytes().hex()


def compute_hashes(coco: dict, cache_path: Path | None = None, workers: int | None = None) -> dict:
    """Return {key: {"sha256", "dhash", "size"}}; cached by (key, file size)."""
    cache = {}
    if cache_path and Path(cache_path).is_file():
        cache = json.loads(Path(cache_path).read_text())
    todo, result = [], {}
    for im in coco["images"]:
        p = image_path(coco, im)
        size = p.stat().st_size
        c = cache.get(im["key"])
        if c and c["size"] == size:
            result[im["key"]] = c
        else:
            todo.append((im["key"], p))
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for key, sha, dh in ex.map(_hash_one, todo, chunksize=16):
                result[key] = {"sha256": sha, "dhash": dh, "size": Path(dict(todo)[key]).stat().st_size}
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(result))
    return result


def cluster_images(hashes: dict, threshold: int = 10) -> tuple[dict, dict]:
    """Return ({key: group_id}, report). group_id is the smallest key in the group."""
    keys = sorted(hashes)
    index = {k: i for i, k in enumerate(keys)}
    parent = list(range(len(keys)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_sha: dict[str, int] = {}
    for k in keys:
        first = by_sha.setdefault(hashes[k]["sha256"], index[k])
        union(first, index[k])

    bits = np.array([np.frombuffer(bytes.fromhex(hashes[k]["dhash"]), dtype=np.uint8) for k in keys])
    for i in range(len(keys) - 1):
        dist = _POPCOUNT[bits[i] ^ bits[i + 1:]].sum(axis=1, dtype=np.int32)
        for j in np.nonzero(dist <= threshold)[0]:
            union(i, i + 1 + int(j))

    members: dict[int, list[str]] = {}
    for k in keys:
        members.setdefault(find(index[k]), []).append(k)
    groups = {k: keys[root] for root, ms in members.items() for k in ms}

    multi = sorted((ms for ms in members.values() if len(ms) > 1), key=len, reverse=True)
    sources = lambda ms: sorted({m.split("/", 1)[0] for m in ms})
    report = {
        "threshold_bits_of_256": threshold,
        "images": len(keys),
        "groups": len(members),
        "images_in_multi_image_groups": sum(len(m) for m in multi),
        "exact_duplicate_files": len(keys) - len(by_sha),
        "cross_source_groups": sum(len(sources(m)) > 1 for m in multi),
        "group_size_histogram": dict(Counter(len(m) for m in multi)),
        "largest_groups": [{"size": len(m), "sources": sources(m), "members": m[:10]} for m in multi[:20]],
    }
    return groups, report
