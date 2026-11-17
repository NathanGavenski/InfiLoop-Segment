import pytest
import torch

from models.encoder import BioCLIP2Encoder, DINOv3Encoder


def _check(enc, h, w):
    x = torch.rand(2, 3, h, w)
    out = enc(x)
    p = enc.patch_size
    assert out.shape == (2, enc.embed_dim, h // p, w // p)
    assert not any(q.requires_grad for q in enc.parameters())
    enc.train()
    assert not enc.training and not enc.model.training
    with pytest.raises(ValueError):
        enc(torch.rand(1, 3, h + 1, w))
    return out


def test_dinov3_wrapper_variable_aspect():
    pytest.importorskip("transformers")
    enc = DINOv3Encoder(dtype="float32", random_init=dict(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
        patch_size=16, num_register_tokens=4))
    _check(enc, 64, 112)
    _check(enc, 96, 48)


def test_bioclip2_wrapper_interpolates_positions():
    open_clip = pytest.importorskip("open_clip")
    vit = open_clip.transformer.VisionTransformer(
        image_size=56, patch_size=14, width=32, layers=1, heads=2, mlp_ratio=2.0, output_dim=16)
    enc = BioCLIP2Encoder(dtype="float32", model=vit)
    assert enc.base_grid == 4
    a = _check(enc, 56, 56)   # native grid, no interpolation
    _check(enc, 42, 98)       # 3x7 grid, interpolated
    assert torch.isfinite(a).all()
