"""Heads over the patch grid. Phase 2 needs the linear probe (B0), the input projection and
the per-token mask head (B1). The halting and presence heads arrive in Phase 4."""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """LayerNorm over channels of a BxCxHxW map (as in the SAM decoder)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class LinearProbe(nn.Module):
    """B0: one linear layer per patch token, encoder dim -> 1 logit."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.norm = LayerNorm2d(in_dim)
        self.linear = nn.Conv2d(in_dim, 1, kernel_size=1)

    def forward(self, feats):
        return self.linear(self.norm(feats))


class InputProjection(nn.Module):
    """Encoder dim -> InfiLoop hidden size (512)."""

    def __init__(self, in_dim: int, hidden: int = 512):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, hidden, kernel_size=1)
        self.norm = LayerNorm2d(hidden)

    def forward(self, feats):
        return self.norm(self.proj(feats))


class MaskHead(nn.Module):
    """Per-token mask logits from the latent. In Phase 3 this reads the answer state."""

    def __init__(self, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(LayerNorm2d(hidden), nn.Conv2d(hidden, hidden // 2, 1),
                                 nn.GELU(), nn.Conv2d(hidden // 2, 1, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
