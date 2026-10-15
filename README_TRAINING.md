# Training on the VM

Local machine = code + CPU smoke tests only. Real runs happen on the GPU VM.

## One-time setup (VM)
```bash
# 1. copy only the code: castus_vision/ without datasets/ and runs/ (configs/splits/splits.json must be included)
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision          # CUDA build matching the VM driver
pip install -r requirements.txt && pip install -e .
nvidia-smi                             # note GPU + VRAM, then pick train.batch / train.workers

# 2. download TACO + BePLi and build the dataset (frozen splits are reused)
python scripts/prepare_dataset.py --raw-root /data/castus_raw
```
`prepare_dataset.py` takes roughly 20-40 min (TACO: ~1500 Flickr images downloaded one at a time, ~7 min, because
Flickr throttles parallel connections; BePLi is a 9.5 GB archive). It is resumable: re-run the same command after an interruption. It aborts if any
image would land in a different split than `configs/splits/splits.json`. `--skip-download` reuses existing raw data.
`dataset_version` may differ from the local one: the VM gets full-resolution TACO images where the local copy has
155 downscaled ones, so mask pixels/hashes differ slightly. Splits are identical.
BePLi terms: non-profit, non-commercial research use only.

## Runs (all use the same TACO val/test images; BePLi val/test are secondary)
```bash
python scripts/train_yolo.py --config configs/train/A_taco_only.yaml        --set train.device=0 train.batch=32 train.workers=8
python scripts/train_yolo.py --config configs/train/B_taco_bepli_plain.yaml --set train.device=0 train.batch=32 train.workers=8
python scripts/train_yolo.py --config configs/train/C_taco_bepli_weighted.yaml --set train.device=0 train.batch=32
python scripts/train_yolo.py --config configs/train/D_taco_bepli_masked.yaml   --set train.device=0 train.batch=32
```
Resume: same command + `--resume`. Re-evaluate: `--eval-only`. Prefer a single GPU for C/D (weighted sampling).
Use the same `train.batch` across A-D so the comparison stays fair.

## Output (`runs/<run_name>/`)
`config.resolved.yaml`, `environment.json` (library versions, dataset_version), `results.csv` + loss/metric plots,
`weights/best.pt|last.pt`, `metrics.json` (val/test on TACO, val/test on BePLi; box+mask mAP, per-class P/R/AP).

## Run ladder
| Run | Train data | Change vs previous |
|---|---|---|
| A | TACO | baseline |
| B | TACO + BePLi | plain mix |
| C | TACO + BePLi | 60/40 source sampling, source-pure mosaic |
| D | TACO + BePLi | C + BCE masked for can/glass/metal_other/paper_cardboard on BePLi background anchors |
