"""Model assembly for the Phase 2 baselines.

B0: frozen encoder -> linear per-patch probe -> bilinear upsampling.
B1: frozen encoder -> projection to 512 -> per-token mask head -> upsampling decoder, one pass.

B1 shares every trainable component with the Phase 3 model except the recursive block,
which will sit between the projection and the mask head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder import MaskDecoder
from models.encoder import FrozenEncoder, build_encoder
from models.heads import InputProjection, LinearProbe, MaskHead


class B0Probe(nn.Module):
    def __init__(self, encoder: FrozenEncoder):
        super().__init__()
        self.encoder = encoder
        self.probe = LinearProbe(encoder.embed_dim)

    def forward_features(self, feats, images):
        coarse = self.probe(feats)
        logits = F.interpolate(coarse, size=images.shape[-2:], mode="bilinear",
                               align_corners=False)
        return {"logits": logits, "coarse": coarse, "loops": 1}

    def forward(self, images):
        return self.forward_features(self.encoder(images), images)


class B1SinglePass(nn.Module):
    def __init__(self, encoder: FrozenEncoder, hidden: int = 512, rgb_guide: bool = True,
                 decoder_channels=(256, 128, 64, 32)):
        super().__init__()
        self.encoder = encoder
        self.proj = InputProjection(encoder.embed_dim, hidden)
        self.mask_head = MaskHead(hidden)
        self.decoder = MaskDecoder(hidden, encoder.patch_size, decoder_channels, rgb_guide)

    def forward_features(self, feats, images):
        z = self.proj(feats)
        coarse = self.mask_head(z)
        logits = self.decoder(z, coarse, images)
        return {"logits": logits, "coarse": coarse, "loops": 1}

    def forward(self, images):
        return self.forward_features(self.encoder(images), images)


def build_model(cfg: dict, encoder: FrozenEncoder | None = None) -> nn.Module:
    encoder = encoder or build_encoder(cfg["encoder"])
    kind = cfg["model"]["kind"]
    if kind == "b0":
        return B0Probe(encoder)
    if kind == "b1":
        m = cfg["model"]
        return B1SinglePass(encoder, m.get("hidden", 512), m.get("rgb_guide", True),
                            tuple(m.get("decoder_channels", (256, 128, 64, 32))))
    raise ValueError(f"unknown model kind {kind!r} (Phase 2 provides b0 and b1)")


def trainable_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]


@torch.no_grad()
def predict_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    return model(images)["logits"][:, 0]
