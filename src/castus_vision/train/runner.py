"""Config-driven YOLO26-seg training + evaluation runner (used by scripts/train_yolo.py).

Everything hardware-dependent (device, batch, workers, cache) lives in the YAML config or CLI overrides.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import yaml

# Keys that are owned by the runner and must not be set under `train:` in a config.
RESERVED = {"data", "model", "project", "name", "resume", "trainer", "task", "mode"}


def load_config(path: Path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    for k in ("run_name", "data", "model", "train"):
        if k not in cfg:
            raise ValueError(f"config missing '{k}'")
    for k in ("root", "train", "val", "test", "sources"):
        if k not in cfg["data"]:
            raise ValueError(f"config data.{k} missing")
    bad = RESERVED & set(cfg["train"])
    if bad:
        raise ValueError(f"train: must not set {sorted(bad)} (managed by the runner)")
    c = cfg.get("castus", {}) or {}
    if c.get("masked_loss") and not c.get("source_pure_mosaic", False):
        raise ValueError("masked_loss requires castus.source_pure_mosaic: true (the loss infers source per image)")
    if c.get("source_pure_mosaic") and (cfg["train"].get("mixup", 0) or cfg["train"].get("cutmix", 0)):
        raise ValueError("source_pure_mosaic needs train.mixup=0 and train.cutmix=0 (they mix images across sources)")


def write_data_yaml(cfg: dict, out: Path, split_lists: dict[str, str]) -> Path:
    """Data YAML pointing at list files. Class names are copied from the exported dataset's data.yaml."""
    root = Path(cfg["data"]["root"]).resolve()
    names = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]
    for fname in set(split_lists.values()):
        if not (root / fname).is_file():
            raise FileNotFoundError(f"{root / fname} missing; run scripts/make_yolo_lists.py")
    doc = {"path": str(root), **split_lists, "names": names}
    out.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return out


def environment_info() -> dict:
    import torch
    import ultralytics

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpus"] = [
            {"name": torch.cuda.get_device_name(i), "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)}
            for i in range(torch.cuda.device_count())
        ]
    return info


def _dataset_version(root: Path) -> str | None:
    p = root / "dataset_info.json"
    return json.loads(p.read_text())["dataset_version"] if p.is_file() else None


def run(cfg: dict, resume: bool = False) -> Path:
    from ultralytics import YOLO

    from castus_vision.train.trainer import ENV_KEY, CastusSegTrainer

    root = Path(cfg["data"]["root"]).resolve()
    project = Path(cfg.get("project", "runs")).resolve()
    run_dir = project / cfg["run_name"]

    opts = dict(cfg.get("castus") or {})
    opts["sources"] = cfg["data"]["sources"]
    os.environ[ENV_KEY] = json.dumps(opts)

    if resume:
        last = run_dir / "weights" / "last.pt"
        if not last.is_file():
            raise FileNotFoundError(f"nothing to resume: {last} not found")
        model = YOLO(str(last))
        model.train(resume=True, trainer=CastusSegTrainer)
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"{run_dir} already exists; choose another run_name or use --resume")
        run_dir.mkdir(parents=True, exist_ok=True)
        data_yaml = write_data_yaml(
            cfg, run_dir / "data.yaml",
            {"train": cfg["data"]["train"], "val": cfg["data"]["val"], "test": cfg["data"]["test"]},
        )
        (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        (run_dir / "environment.json").write_text(
            json.dumps({**environment_info(), "dataset_version": _dataset_version(root)}, indent=2), encoding="utf-8"
        )
        model = YOLO(cfg["model"])
        model.train(
            data=str(data_yaml), project=str(project), name=cfg["run_name"], exist_ok=True,
            trainer=CastusSegTrainer, **cfg["train"],
        )

    return evaluate(cfg, run_dir)


def evaluate(cfg: dict, run_dir: Path) -> Path:
    """Evaluate best.pt on the configured val/test lists (+ extra per-source lists). Writes metrics.json."""
    from ultralytics import YOLO

    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"{best} not found")
    model = YOLO(str(best))
    d, t = cfg["data"], cfg["train"]
    sets = {"val": d["val"], "test": d["test"], **(d.get("extra_eval") or {})}
    eval_args = {k: t[k] for k in ("imgsz", "batch", "device", "workers") if t.get(k) is not None}
    report: dict = {"weights": str(best), "eval_args": eval_args, "sets": {}}
    for name, lst in sets.items():
        data_yaml = write_data_yaml(cfg, run_dir / f"eval_{name}.yaml", {"train": lst, "val": lst, "test": lst})
        m = model.val(data=str(data_yaml), split="val", plots=False, project=str(run_dir / "eval"),
                      name=name, exist_ok=True, **eval_args)
        names = m.names
        per_class = {}
        for pos, cid in enumerate(m.box.ap_class_index):
            bp, br, bap50, bap = m.box.class_result(pos)
            mp, mr, map50, mapc = m.seg.class_result(pos)
            per_class[names[int(cid)]] = {
                "box": {"precision": bp, "recall": br, "ap50": bap50, "ap50_95": bap},
                "mask": {"precision": mp, "recall": mr, "ap50": map50, "ap50_95": mapc},
            }
        report["sets"][name] = {
            "list": lst,
            "box_map50_95": float(m.box.map), "box_map50": float(m.box.map50),
            "mask_map50_95": float(m.seg.map), "mask_map50": float(m.seg.map50),
            "per_class": per_class,
            "speed_ms_per_image": m.speed,
        }
    out = run_dir / "metrics.json"
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return out
