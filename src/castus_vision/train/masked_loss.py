"""Masked classification loss: do not penalise "background" predictions for classes a source cannot annotate.

Motivation: BePLi only labels plastic litter. In a BePLi image, an unlabelled can or glass bottle is not a real
negative, so the standard BCE would teach the model to suppress it. We zero the BCE term only for

    (image from a masked source) AND (anchor assigned to no object) AND (class in masked_classes)

Anchors assigned to a labelled object still get negative supervision for other classes, and positives are never
masked. TACO images (fully annotated for all 7 classes) are untouched.
"""
from __future__ import annotations

import torch
from torch import nn


class SourceState:
    """Per-batch flags shared between the trainer and every MaskedBCE instance."""

    def __init__(self) -> None:
        self.flags: torch.Tensor | None = None  # (batch,) bool, True = image comes from a masked source


class MaskedBCE(nn.Module):
    def __init__(self, base: nn.Module, state: SourceState, masked_class_ids: list[int], nc: int) -> None:
        super().__init__()
        if getattr(base, "reduction", None) != "none":
            raise ValueError("MaskedBCE needs an elementwise (reduction='none') base loss")
        self.base = base
        self.state = state
        mask = torch.zeros(nc, dtype=torch.bool)
        mask[list(masked_class_ids)] = True
        self.class_mask = mask

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.base(pred, target)  # (bs, anchors, nc)
        flags = self.state.flags
        if flags is None or flags.numel() != loss.shape[0] or not bool(flags.any()):
            return loss
        cm = self.class_mask.to(loss.device)
        background = target.sum(-1, keepdim=True) == 0  # (bs, anchors, 1)
        drop = flags.to(loss.device).view(-1, 1, 1) & background & cm.view(1, 1, -1)
        return loss * (~drop).to(loss.dtype)


def install_masked_bce(criterion, state: SourceState, masked_class_ids: list[int], nc: int) -> int:
    """Patch the BCE of every loss object inside an Ultralytics criterion. Returns the number patched."""
    loss_objs = [getattr(criterion, "one2many", None), getattr(criterion, "one2one", None)]
    loss_objs = [o for o in loss_objs if o is not None] or [criterion]
    n = 0
    for obj in loss_objs:
        if hasattr(obj, "bce") and not isinstance(obj.bce, MaskedBCE):
            obj.bce = MaskedBCE(obj.bce, state, masked_class_ids, nc)
            n += 1
    if n == 0:
        raise RuntimeError("no loss object with a .bce attribute found; Ultralytics internals changed?")
    return n
