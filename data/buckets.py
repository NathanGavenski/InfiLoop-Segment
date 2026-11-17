"""Long-side resizing, padding to the patch size, padding masks and aspect-ratio buckets.

Every frame is resized so its long side equals `long_side` with the aspect ratio kept, then
padded on the bottom and right to a multiple of the patch size. The padding mask is 1 on real
pixels and 0 on padding, and the loss and metrics ignore padded pixels. Frames whose padded
size matches share a bucket, so a batch never mixes shapes.

Encoders with a fixed input size (SAM 3 at 1008 px) use `square=True`, which pads every frame
to long_side x long_side and therefore puts every frame in one bucket.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Hashable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def resized_hw(height: int, width: int, long_side: int) -> tuple[int, int]:
    """Size after scaling the long side to `long_side`, aspect ratio kept."""
    scale = long_side / max(height, width)
    if height >= width:
        return long_side, max(1, round(width * scale))
    return max(1, round(height * scale)), long_side


def padded_hw(height: int, width: int, patch: int, square: bool = False) -> tuple[int, int]:
    """Smallest size >= (height, width) that is a multiple of `patch` on both sides."""
    if square:
        side = math.ceil(max(height, width) / patch) * patch
        return side, side
    return math.ceil(height / patch) * patch, math.ceil(width / patch) * patch


def bucket_key(height: int, width: int, long_side: int, patch: int,
               square: bool = False) -> tuple[int, int]:
    """Bucket of an original frame size: its padded size after resizing."""
    return padded_hw(*resized_hw(height, width, long_side), patch, square)


def resize_pad(image: torch.Tensor, mask: torch.Tensor | None, long_side: int, patch: int,
               square: bool = False, target_hw: tuple[int, int] | None = None):
    """Resize a CxHxW float image (and an HxW mask) by long side and pad to the patch grid.

    Returns (image, mask, valid) where valid is the HxW padding mask (1 = real pixel).
    The image is resized bilinearly with antialiasing and the mask with nearest-exact, so
    mask values stay binary. `target_hw` forces the resized size, which keeps a cropped frame
    in the bucket of its uncropped original.
    """
    _, h, w = image.shape
    nh, nw = target_hw or resized_hw(h, w, long_side)
    ph, pw = padded_hw(nh, nw, patch, square)
    if (nh, nw) != (h, w):
        image = F.interpolate(image[None], size=(nh, nw), mode="bilinear",
                              align_corners=False, antialias=True)[0]
    image = F.pad(image, (0, pw - nw, 0, ph - nh))
    valid = torch.zeros(ph, pw, dtype=torch.bool)
    valid[:nh, :nw] = True
    if mask is not None:
        if (nh, nw) != (h, w):
            mask = F.interpolate(mask[None, None].float(), size=(nh, nw),
                                 mode="nearest-exact")[0, 0]
        mask = F.pad(mask.float(), (0, pw - nw, 0, ph - nh))
    return image, mask, valid


def unpad_resize(logits: torch.Tensor, resized: tuple[int, int],
                 original: tuple[int, int]) -> torch.Tensor:
    """Crop padded ...xHxW logits to the resized region and resize back to the original."""
    nh, nw = resized
    x = logits[..., :nh, :nw]
    lead = x.shape[:-2]
    x = x.reshape(-1, 1, nh, nw).float()
    x = F.interpolate(x, size=original, mode="bilinear", align_corners=False)
    return x.reshape(*lead, *original)


def make_bucket_batches(indices: Sequence[int], keys: Sequence[Hashable], batch_size: int,
                        rng: np.random.Generator | None = None,
                        drop_last: bool = False) -> list[list[int]]:
    """Group `indices` into batches whose `keys[i]` all match.

    With an rng, items are shuffled inside each bucket and the batch order is shuffled too.
    """
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for i in indices:
        groups[keys[i]].append(int(i))
    batches = []
    for key in sorted(groups, key=str):
        items = groups[key]
        if rng is not None:
            rng.shuffle(items)
        for s in range(0, len(items), batch_size):
            b = items[s:s + batch_size]
            if drop_last and len(b) < batch_size:
                continue
            batches.append(b)
    if rng is not None:
        order = rng.permutation(len(batches))
        batches = [batches[i] for i in order]
    return batches
