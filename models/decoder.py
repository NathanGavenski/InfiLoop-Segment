"""Light upsampling mask decoder in the style of the SAM mask decoder.

It turns per-token mask logits (plus the latent they came from) into a full-resolution mask.
Each x2 stage is a transposed convolution, and the downsampled RGB frame is concatenated in
as a guide at every scale, which helps recover thin legs, tails and ears. The decoder
predicts a residual on top of the bilinearly upsampled coarse logits, so a zero-initialised
decoder starts from plain bilinear upsampling.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.heads import LayerNorm2d


class UpStage(nn.Module):
    def __init__(self, c_in: int, c_out: int, guide_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2)
        self.norm = LayerNorm2d(c_out)
        self.guide = nn.Conv2d(3, guide_ch, 3, padding=1) if guide_ch else None
        self.fuse = nn.Sequential(nn.Conv2d(c_out + guide_ch, c_out, 3, padding=1), nn.GELU())

    def forward(self, x, image):
        x = F.gelu(self.norm(self.up(x)))
        if self.guide is not None:
            g = F.interpolate(image, size=x.shape[-2:], mode="bilinear", align_corners=False,
                              antialias=True)
            x = torch.cat([x, F.gelu(self.guide(g))], dim=1)
        return self.fuse(x)


class MaskDecoder(nn.Module):
    def __init__(self, hidden: int = 512, patch_size: int = 16, channels=(256, 128, 64, 32),
                 rgb_guide: bool = True, guide_ch: int = 16):
        super().__init__()
        n_up = math.ceil(math.log2(patch_size))
        channels = list(channels)[:n_up] + [channels[-1]] * max(0, n_up - len(channels))
        self.inp = nn.Conv2d(hidden + 1, channels[0], 1)
        ins = [channels[0]] + channels[:-1]
        self.stages = nn.ModuleList(UpStage(i, o, guide_ch if rgb_guide else 0)
                                    for i, o in zip(ins, channels))
        self.out = nn.Conv2d(channels[-1], 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z: torch.Tensor, coarse: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """z: BxDxhxw latent, coarse: Bx1xhxw logits, image: Bx3xHxW in [0, 1]."""
        size = image.shape[-2:]
        x = self.inp(torch.cat([z, coarse], dim=1))
        for stage in self.stages:
            x = stage(x, image)
        residual = F.interpolate(self.out(x), size=size, mode="bilinear", align_corners=False)
        base = F.interpolate(coarse, size=size, mode="bilinear", align_corners=False)
        return base + residual
