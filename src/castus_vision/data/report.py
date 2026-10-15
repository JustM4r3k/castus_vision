"""Dataset composition report: classes, sources and object sizes per split."""
from __future__ import annotations

import math
from collections import Counter

from castus_vision.data.split import SPLITS

# Object size = sqrt(mask area) after the image is resized so its long side equals imgsz.
SIZE_BINS = [("<8px", 0, 8), ("8-16px", 8, 16), ("16-32px", 16, 32), ("32-96px", 32, 96), (">=96px", 96, math.inf)]


def size_bin(sqrt_area: float) -> str:
    for name, lo, hi in SIZE_BINS:
        if lo <= sqrt_area < hi:
            return name
    return SIZE_BINS[-1][0]


def composition_report(coco: dict, assignment: dict, excluded: dict, imgsz: int = 640) -> dict:
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}
    per = {s: {"images": 0, "instances": 0, "classes": Counter(), "sources_images": Counter(),
               "sources_instances": Counter(), "size_bins": Counter(),
               "size_bins_by_class": {n: Counter() for n in cats.values()}} for s in SPLITS}
    for im in coco["images"]:
        s = assignment.get(im["key"])
        if s:
            per[s]["images"] += 1
            per[s]["sources_images"][im["source"]] += 1
    for a in coco["annotations"]:
        im = images[a["image_id"]]
        s = assignment.get(im["key"])
        if not s:
            continue
        b = size_bin(math.sqrt(a["area"]) * imgsz / max(im["width"], im["height"]))
        p = per[s]
        p["instances"] += 1
        p["classes"][cats[a["category_id"]]] += 1
        p["sources_instances"][im["source"]] += 1
        p["size_bins"][b] += 1
        p["size_bins_by_class"][cats[a["category_id"]]][b] += 1
    out = {"imgsz_for_size_bins": imgsz,
           "excluded_images": dict(Counter(excluded.values())),
           "splits": {}}
    for s, p in per.items():
        out["splits"][s] = {
            "images": p["images"], "instances": p["instances"],
            "classes": {n: p["classes"][n] for n in cats.values()},
            "sources_images": dict(p["sources_images"]),
            "sources_instances": dict(p["sources_instances"]),
            "size_bins": {n: p["size_bins"][n] for n, _, _ in SIZE_BINS},
            "size_bins_by_class": {c: {n: v[n] for n, _, _ in SIZE_BINS} for c, v in p["size_bins_by_class"].items()},
        }
    return out


def report_markdown(report: dict, extra: dict | None = None) -> str:
    lines = ["# Dataset composition", ""]
    splits = report["splits"]
    lines += ["## Images and instances", "", "| split | images | instances |", "|---|---|---|"]
    lines += [f"| {s} | {v['images']} | {v['instances']} |" for s, v in splits.items()]
    classes = list(next(iter(splits.values()))["classes"])
    lines += ["", "## Instances per class", "", "| class | " + " | ".join(splits) + " |", "|---|" + "---|" * len(splits)]
    lines += [f"| {c} | " + " | ".join(str(splits[s]["classes"][c]) for s in splits) + " |" for c in classes]
    bins = list(next(iter(splits.values()))["size_bins"])
    lines += ["", f"## Object size (sqrt of mask area at long side = {report['imgsz_for_size_bins']}px)", "",
              "| bin | " + " | ".join(splits) + " |", "|---|" + "---|" * len(splits)]
    lines += [f"| {b} | " + " | ".join(str(splits[s]["size_bins"][b]) for s in splits) + " |" for b in bins]
    sources = sorted({src for v in splits.values() for src in v["sources_images"]})
    lines += ["", "## Images per source", "", "| source | " + " | ".join(splits) + " |", "|---|" + "---|" * len(splits)]
    lines += [f"| {src} | " + " | ".join(str(splits[s]["sources_images"].get(src, 0)) for s in splits) + " |" for src in sources]
    lines += ["", "## Excluded images", ""]
    lines += [f"- {k}: {v}" for k, v in report["excluded_images"].items()] or ["- none"]
    for title, data in (extra or {}).items():
        lines += ["", f"## {title}", "", "```", str(data), "```"]
    return "\n".join(lines) + "\n"
