"""Frozen image encoders that return a 2D grid of patch tokens.

All encoders take a Bx3xHxW batch in [0, 1] whose sides are multiples of the patch size
(data/buckets.py guarantees this), apply their own normalisation, run in bf16 under
no_grad, and return BxCx(H/p)x(W/p) float32 features. Parameters never require grad and
the module stays in eval mode even when the parent model calls .train().

Config (the `encoder:` block of an experiment YAML)::

    encoder:
      name: dinov3            # dinov3 | bioclip2 | sam3
      model_id: facebook/dinov3-vitl16-pretrain-lvd1689m
      dtype: bfloat16

Licences differ per encoder (DINOv3 License, BioCLIP 2 model card, SAM License); check
them before choosing one (PLAN.md Phase 2).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class FrozenEncoder(nn.Module):
    """Base class: normalisation, freezing, dtype handling."""

    patch_size: int
    embed_dim: int
    fixed_size: int | None = None  # input side the encoder requires, if any

    def __init__(self, mean, std, dtype: str = "bfloat16"):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.compute_dtype = DTYPES[dtype]

    def freeze(self) -> "FrozenEncoder":
        for p in self.parameters():
            p.requires_grad_(False)
        return super().train(False)

    def train(self, mode: bool = True) -> "FrozenEncoder":
        return super().train(False)  # always eval: no dropout, frozen norm statistics

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised input -> BxNxC patch tokens in raster order. Implemented per encoder."""
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, _, h, w = images.shape
        p = self.patch_size
        if h % p or w % p:
            raise ValueError(f"input {h}x{w} is not a multiple of patch size {p}")
        if self.fixed_size and (h, w) != (self.fixed_size, self.fixed_size):
            raise ValueError(f"{type(self).__name__} needs {self.fixed_size}x{self.fixed_size} "
                             "inputs; set data.long_side to it and data.square: true")
        x = (images - self.mean) / self.std
        use_amp = images.is_cuda and self.compute_dtype != torch.float32
        with torch.autocast(images.device.type, dtype=self.compute_dtype, enabled=use_amp):
            t = self.tokens(x)
        gh, gw = h // p, w // p
        return t.float().transpose(1, 2).reshape(b, -1, gh, gw)


class DINOv3Encoder(FrozenEncoder):
    """DINOv3 ViT via transformers. RoPE positions, so any multiple of 16 works."""

    def __init__(self, model_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
                 dtype: str = "bfloat16", model: nn.Module | None = None,
                 random_init: dict | None = None):
        super().__init__(IMAGENET_MEAN, IMAGENET_STD, dtype)
        if model is None and random_init is not None:
            # Tiny randomly initialised ViT for smoke tests; never for reported results.
            from transformers import DINOv3ViTConfig, DINOv3ViTModel
            model = DINOv3ViTModel(DINOv3ViTConfig(**random_init))
        elif model is None:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_id)
        self.model = model
        cfg = model.config
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.hidden_size
        self.n_prefix = 1 + getattr(cfg, "num_register_tokens", 0)
        self.freeze()

    def tokens(self, x):
        out = self.model(pixel_values=x).last_hidden_state  # already layer-normed
        return out[:, self.n_prefix:]


class BioCLIP2Encoder(FrozenEncoder):
    """BioCLIP 2 (open_clip ViT-L/14). Learned 2D position embeddings are bicubically
    interpolated to the input grid so frames keep their aspect ratio."""

    def __init__(self, model_id: str = "hf-hub:imageomics/bioclip-2", dtype: str = "bfloat16",
                 model: nn.Module | None = None):
        super().__init__(CLIP_MEAN, CLIP_STD, dtype)
        if model is None:
            import open_clip
            clip, _, _ = open_clip.create_model_and_transforms(model_id)
            model = clip.visual
        self.model = model
        self.patch_size = model.patch_size[0]
        self.embed_dim = model.conv1.out_channels
        n = model.positional_embedding.shape[0] - 1
        self.base_grid = int(math.isqrt(n))
        self.freeze()

    def _pos(self, gh: int, gw: int) -> torch.Tensor:
        pos = self.model.positional_embedding
        cls, grid = pos[:1], pos[1:]
        if (gh, gw) != (self.base_grid, self.base_grid):
            g = grid.reshape(1, self.base_grid, self.base_grid, -1).permute(0, 3, 1, 2)
            g = F.interpolate(g.float(), size=(gh, gw), mode="bicubic", align_corners=False)
            grid = g.permute(0, 2, 3, 1).reshape(gh * gw, -1).to(pos.dtype)
        return torch.cat([cls, grid])

    def tokens(self, x):
        m = self.model
        x = m.conv1(x)
        gh, gw = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        cls = m.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1) + self._pos(gh, gw).to(x.dtype)
        x = m.ln_pre(x)
        x = m.transformer(x)
        return m.ln_post(x)[:, 1:]


class SAM3Encoder(FrozenEncoder):
    """SAM 3 perception encoder, fixed at 1008x1008 (stride-14 grid of 72x72).

    `backbone.forward_image` returns the neck's stride-14 map as `vision_features` (checked
    against the pinned sam3 commit); not yet run with the real weights.
    """

    fixed_size = 1008

    def __init__(self, checkpoint: str | None = None, dtype: str = "bfloat16",
                 model: nn.Module | None = None, **_):
        super().__init__((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), dtype)
        if model is None:
            from sam3.model_builder import build_sam3_image_model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = build_sam3_image_model(checkpoint_path=checkpoint, device=device)
        self.backbone = model.backbone
        self.patch_size = 14
        self.embed_dim = 256
        self.freeze()

    def tokens(self, x):
        out = self.backbone.forward_image(x)
        feats = out["vision_features"] if isinstance(out, dict) else out
        return feats.flatten(2).transpose(1, 2)


ENCODERS = {"dinov3": DINOv3Encoder, "bioclip2": BioCLIP2Encoder, "sam3": SAM3Encoder}


def build_encoder(cfg: dict) -> FrozenEncoder:
    cfg = dict(cfg)
    name = cfg.pop("name", "dinov3")
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder {name!r}; choose from {sorted(ENCODERS)}")
    return ENCODERS[name](**cfg)
