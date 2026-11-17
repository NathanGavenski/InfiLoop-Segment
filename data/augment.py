"""Training augmentation from PLAN.md Phase 1: horizontal flips, mild colour jitter, random
greyscale (to imitate infrared night frames) and scale crops centred on small animals.

The small-animal crop runs on the original frame and keeps its aspect ratio, so the frame
stays in the same aspect-ratio bucket after resizing. The photometric steps and the flip run
on the resized CxHxW float image in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class AugmentConfig:
    hflip: float = 0.5
    brightness: float = 0.15
    contrast: float = 0.15
    saturation: float = 0.15
    greyscale: float = 0.1
    small_crop: float = 0.3          # probability of a crop centred on a small instance
    small_area: float = 96 ** 2      # instances below this (original px) count as small
    crop_scale: tuple[float, float] = (0.35, 0.7)  # crop side as a fraction of the frame


def small_animal_crop(image: np.ndarray, mask: np.ndarray, boxes: list, areas: list,
                      cfg: AugmentConfig, rng: np.random.Generator):
    """Crop an HxWxC image and its HxW mask around a random small instance.

    Returns the inputs unchanged when the frame has no small instance or the draw misses.
    """
    small = [b for b, a in zip(boxes, areas) if a < cfg.small_area]
    if not small or rng.random() >= cfg.small_crop:
        return image, mask
    h, w = mask.shape
    x, y, bw, bh = small[rng.integers(len(small))]
    # One scale for both sides keeps the aspect ratio; grow it so the whole box fits.
    s = min(1.0, max(rng.uniform(*cfg.crop_scale), (bh + 2) / h, (bw + 2) / w))
    ch, cw = max(round(h * s), 1), max(round(w * s), 1)
    cx = x + bw / 2 + rng.uniform(-0.25, 0.25) * cw
    cy = y + bh / 2 + rng.uniform(-0.25, 0.25) * ch
    x0 = int(np.clip(cx - cw / 2, 0, w - cw))
    y0 = int(np.clip(cy - ch / 2, 0, h - ch))
    return image[y0:y0 + ch, x0:x0 + cw], mask[y0:y0 + ch, x0:x0 + cw]


def photometric_and_flip(image: torch.Tensor, mask: torch.Tensor, cfg: AugmentConfig,
                         rng: np.random.Generator):
    """Colour jitter, greyscale and horizontal flip on a CxHxW image in [0, 1]."""
    if cfg.brightness:
        image = image * rng.uniform(1 - cfg.brightness, 1 + cfg.brightness)
    if cfg.contrast:
        mean = image.mean()
        image = (image - mean) * rng.uniform(1 - cfg.contrast, 1 + cfg.contrast) + mean
    grey = (0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2])[None]
    if cfg.saturation:
        image = (image - grey) * rng.uniform(1 - cfg.saturation, 1 + cfg.saturation) + grey
    if rng.random() < cfg.greyscale:
        image = grey.expand(3, -1, -1).clone()
    if rng.random() < cfg.hflip:
        image = image.flip(-1)
        mask = mask.flip(-1)
    return image.clamp_(0, 1), mask
