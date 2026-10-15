"""Train and evaluate YOLO26-seg from a config.

    python scripts/train_yolo.py --config configs/train/A_taco_only.yaml
    python scripts/train_yolo.py --config configs/train/A_taco_only.yaml --set train.batch=32 train.device=0
    python scripts/train_yolo.py --config configs/train/A_taco_only.yaml --resume
    python scripts/train_yolo.py --config configs/train/A_taco_only.yaml --eval-only
"""
import argparse
from pathlib import Path

from castus_vision.train.runner import evaluate, load_config, run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="section.key=value", help="override config values")
    p.add_argument("--resume", action="store_true", help="resume from runs/<run_name>/weights/last.pt")
    p.add_argument("--eval-only", action="store_true", help="skip training; evaluate existing best.pt")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    if a.eval_only:
        run_dir = Path(cfg.get("project", "runs")).resolve() / cfg["run_name"]
        print(f"metrics written to {evaluate(cfg, run_dir)}")
    else:
        print(f"metrics written to {run(cfg, resume=a.resume)}")


if __name__ == "__main__":
    main()
