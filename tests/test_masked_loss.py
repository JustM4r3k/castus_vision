import torch
from torch import nn

from castus_vision.train.masked_loss import MaskedBCE, SourceState


def _make(masked_ids, flags):
    st = SourceState()
    st.flags = torch.tensor(flags)
    return MaskedBCE(nn.BCEWithLogitsLoss(reduction="none"), st, masked_ids, nc=4), st


def test_masks_only_background_negatives_of_masked_classes_in_masked_source():
    m, _ = _make([0, 2], [True, False])
    pred = torch.zeros(2, 3, 4)
    target = torch.zeros(2, 3, 4)
    target[:, 1, 1] = 1.0  # anchor 1 is assigned to class 1 in both images
    out = m(pred, target)
    base = nn.BCEWithLogitsLoss(reduction="none")(pred, target)
    # image 0 (masked source): background anchors 0 and 2, classes 0 and 2 -> zero
    assert out[0, 0, 0] == 0 and out[0, 0, 2] == 0 and out[0, 2, 0] == 0
    # unmasked classes in background anchors keep their loss
    assert out[0, 0, 1] == base[0, 0, 1] and out[0, 0, 3] == base[0, 0, 3]
    # the assigned anchor keeps all terms (positive and negatives of other classes)
    assert torch.equal(out[0, 1], base[0, 1])
    # image 1 (other source) is untouched
    assert torch.equal(out[1], base[1])


def test_noop_without_flags():
    m, st = _make([0], [True])
    st.flags = None
    pred, target = torch.randn(1, 2, 4), torch.zeros(1, 2, 4)
    assert torch.equal(m(pred, target), nn.BCEWithLogitsLoss(reduction="none")(pred, target))
