"""Source-aware training data: weighted source sampling (run C) and source-pure mosaics (needed for run D).

Implemented by swapping the class of the dataset / Mosaic transform that Ultralytics already built, so every other
behaviour (caching, augmentation, label handling) stays exactly as shipped. Tested against ultralytics 8.4.x.
"""
from __future__ import annotations

import random
from pathlib import Path

from ultralytics.data.augment import Mosaic
from ultralytics.data.dataset import YOLODataset


def source_of(im_file: str, sources: list[str]) -> str:
    """Source name from the exported file name prefix (`<source>_...`)."""
    name = Path(im_file).name
    for s in sources:
        if name.startswith(f"{s}_"):
            return s
    return "unknown"


class SourcePureMosaic(Mosaic):
    """Mosaic whose extra tiles come from the same source as the parent image.

    Keeps the per-image `im_file` (which identifies the source) valid for every tile of the mosaic, which the
    masked loss depends on.
    """

    def get_params(self, labels):
        self._parent_source = self.dataset.sources[self.dataset.index_of(labels["im_file"])]
        return super().get_params(labels)

    def get_indexes(self):
        pool = self.dataset.pools[self._parent_source]
        return random.choices(pool, k=self.n - 1)


class SourceAwareYOLODataset(YOLODataset):
    """YOLODataset with optional weighted source sampling. Setup via `configure_sources`."""

    def configure_sources(self, sources: list[str], sampling: dict[str, float] | None, pure_mosaic: bool) -> None:
        self.source_names = sources
        self.sources = [source_of(lb["im_file"], sources) for lb in self.labels]
        self.pools: dict[str, list[int]] = {}
        for i, s in enumerate(self.sources):
            self.pools.setdefault(s, []).append(i)
        self._file_index = {lb["im_file"]: i for i, lb in enumerate(self.labels)}
        if "unknown" in self.pools:
            raise ValueError(f"{len(self.pools['unknown'])} training images have no known source prefix {sources}")

        self.sampling = None
        if sampling:
            missing = [s for s in sampling if s not in self.pools]
            if missing:
                raise ValueError(f"sampling sources {missing} have no images in the training list")
            total = sum(sampling.values())
            if total <= 0:
                raise ValueError("sampling fractions must sum to > 0")
            self.sampling = {s: w / total for s, w in sampling.items()}

        if pure_mosaic:
            self.transforms = self.build_transforms(self._hyp)
            _swap_mosaic(self.transforms)

    def index_of(self, im_file: str) -> int:
        return self._file_index[im_file]

    def __getitem__(self, index):
        if self.sampling:
            src = random.choices(list(self.sampling), weights=list(self.sampling.values()))[0]
            index = random.choice(self.pools[src])
        return super().__getitem__(index)


def _swap_mosaic(compose) -> int:
    """Replace every Mosaic inside a (possibly nested) Compose with SourcePureMosaic. Returns how many."""
    n = 0
    for t in getattr(compose, "transforms", []):
        if type(t) is Mosaic:
            t.__class__ = SourcePureMosaic
            n += 1
        else:
            n += _swap_mosaic(t)
    return n


def make_source_aware(dataset: YOLODataset, hyp, sources: list[str], sampling, pure_mosaic: bool) -> YOLODataset:
    if type(dataset) is not YOLODataset:
        raise TypeError(f"expected a plain YOLODataset, got {type(dataset).__name__}")
    dataset.__class__ = SourceAwareYOLODataset
    dataset._hyp = hyp
    dataset.configure_sources(sources, sampling, pure_mosaic)
    return dataset
