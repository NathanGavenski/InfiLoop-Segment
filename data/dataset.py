"""Torch dataset over a frame table (data/sa_fari.py output). Stage B reuses it unchanged.

Each item is resized by long side, padded to the patch grid and returned with its padding
mask. Images stay in [0, 1]; the encoder wrapper applies its own normalisation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.augment import AugmentConfig, photometric_and_flip, small_animal_crop
from data.buckets import bucket_key, resize_pad, resized_hw
from data.sa_fari import decode


class FrameDataset(Dataset):
    def __init__(self, frames: pd.DataFrame, image_root: str | Path, long_side: int,
                 patch: int, square: bool = False, augment: AugmentConfig | None = None,
                 seed: int = 0):
        self.frames = frames.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.long_side, self.patch, self.square = long_side, patch, square
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.keys = [bucket_key(h, w, long_side, patch, square)
                     for h, w in zip(self.frames["height"], self.frames["width"])]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.frames)

    def load_image(self, i: int) -> np.ndarray:
        row = self.frames.iloc[i]
        with Image.open(self.image_root / row["file_name"]) as im:
            img = np.array(im.convert("RGB"))
        if img.shape[:2] != (row["height"], row["width"]):
            raise ValueError(f"{row['file_name']}: image {img.shape[:2]} but annotation "
                             f"{row['height']}x{row['width']}")
        return img

    def __getitem__(self, i: int) -> dict:
        row = self.frames.iloc[i]
        h, w = int(row["height"]), int(row["width"])
        img = self.load_image(i)
        mask = decode(row.get("union_rle"), h, w)
        rng = np.random.default_rng((self.seed, self.epoch, i))
        if self.augment is not None:
            img, mask = small_animal_crop(img, mask, _as_list(row.get("inst_bboxes")),
                                          _as_list(row.get("inst_areas")), self.augment, rng)
        image = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255
        target = torch.from_numpy(np.ascontiguousarray(mask))
        resized = resized_hw(h, w, self.long_side)
        image, target, valid = resize_pad(image, target, self.long_side, self.patch,
                                          self.square, target_hw=resized)
        if self.augment is not None:
            nh, nw = resized
            img_r, tgt_r = photometric_and_flip(image[:, :nh, :nw], target[:nh, :nw],
                                                self.augment, rng)
            image[:, :nh, :nw], target[:nh, :nw] = img_r, tgt_r
        return {
            "image": image, "mask": target, "valid": valid,
            "presence": torch.tensor(float(row["presence"])),
            "index": i, "resized_hw": torch.tensor(resized), "orig_hw": torch.tensor((h, w)),
        }


def _as_list(x) -> list:
    """Parquet list columns come back as numpy arrays (of arrays); None when missing."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    return [v.tolist() if isinstance(v, np.ndarray) else v for v in x]


def collate(batch: list[dict]) -> dict:
    out = {}
    for k in batch[0]:
        v = [b[k] for b in batch]
        out[k] = torch.stack(v) if isinstance(v[0], torch.Tensor) else torch.tensor(v)
    return out
