"""Custom Ultralytics segmentation trainer for the A/B/C/D experiment ladder.

Options travel through the CASTUS_TRAIN_OPTIONS environment variable (JSON) so they also reach DDP subprocesses.
Keys (all optional):
    sources:             list of source prefixes, e.g. ["taco", "bepli"]
    sampling:            {source: weight}  -> weighted source sampling per training sample (run C)
    source_pure_mosaic:  bool              -> mosaic tiles only from the same source
    masked_loss:         {source: str, classes: [names]}  -> run D
"""
from __future__ import annotations

import json
import os

import torch
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model

from castus_vision.train.masked_loss import SourceState, install_masked_bce
from castus_vision.train.sampling import make_source_aware, source_of

ENV_KEY = "CASTUS_TRAIN_OPTIONS"


def get_options() -> dict:
    return json.loads(os.environ.get(ENV_KEY, "{}"))


class CastusSegTrainer(SegmentationTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opts = get_options()
        self.source_state = SourceState()
        if self.opts.get("masked_loss"):
            self.add_callback("on_train_start", _install_masked_loss)

    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        sampling, pure = self.opts.get("sampling"), bool(self.opts.get("source_pure_mosaic"))
        if mode == "train" and (sampling or pure):
            dataset = make_source_aware(dataset, self.args, self.opts["sources"], sampling, pure)
            LOGGER.info(f"CASTUS dataset: sources={ {s: len(p) for s, p in dataset.pools.items()} } "
                        f"sampling={dataset.sampling} pure_mosaic={pure}")
        return dataset

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        ml = self.opts.get("masked_loss")
        if ml:
            sources = self.opts["sources"]
            batch_src = [source_of(f, sources) for f in batch["im_file"]]
            self.source_state.flags = torch.tensor([s == ml["source"] for s in batch_src], dtype=torch.bool)
        return batch


def _install_masked_loss(trainer: CastusSegTrainer) -> None:
    ml = trainer.opts["masked_loss"]
    names = {v: k for k, v in trainer.data["names"].items()}
    unknown = [c for c in ml["classes"] if c not in names]
    if unknown:
        raise ValueError(f"masked_loss.classes not in dataset names: {unknown}")
    ids = [names[c] for c in ml["classes"]]
    model = unwrap_model(trainer.model)
    if getattr(model, "criterion", None) is None:  # on resume Ultralytics has already rebuilt it
        model.criterion = model.init_criterion()
    n = install_masked_bce(model.criterion, trainer.source_state, ids, len(trainer.data["names"]))
    LOGGER.info(f"CASTUS masked loss active: source={ml['source']} classes={ml['classes']} ({n} loss heads patched)")
