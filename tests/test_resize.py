import numpy as np
import pytest
import torch

from data.buckets import (bucket_key, make_bucket_batches, padded_hw, resize_pad, resized_hw,
                          unpad_resize)


@pytest.mark.parametrize("h,w", [(720, 1280), (1080, 1920), (240, 320), (1520, 2688),
                                 (1280, 720), (194, 320), (500, 500)])
@pytest.mark.parametrize("long_side", [768, 1024, 1344])
def test_long_side_and_aspect(h, w, long_side):
    nh, nw = resized_hw(h, w, long_side)
    assert max(nh, nw) == long_side
    assert abs(nh / nw - h / w) <= 1 / min(nh, nw)  # within one pixel of rounding


@pytest.mark.parametrize("patch", [14, 16])
def test_padded_is_smallest_multiple(patch):
    for h, w in [(576, 1024), (1024, 1024), (1, 1), (17, 33)]:
        ph, pw = padded_hw(h, w, patch)
        assert ph % patch == 0 and pw % patch == 0
        assert h <= ph < h + patch and w <= pw < w + patch
    assert padded_hw(576, 1024, patch, square=True)[0] == padded_hw(576, 1024, patch, square=True)[1]


def test_resize_pad_shapes_valid_mask_and_binary_mask():
    img = torch.rand(3, 720, 1280)
    mask = torch.zeros(720, 1280)
    mask[100:300, 200:500] = 1
    out, m, valid = resize_pad(img, mask, long_side=1000, patch=16)
    nh, nw = resized_hw(720, 1280, 1000)
    assert out.shape == (3, *padded_hw(nh, nw, 16)) == (3, 576, 1008)
    assert m.shape == valid.shape == out.shape[1:]
    assert valid.sum() == nh * nw and valid[:nh, :nw].all()
    assert (out[:, nh:, :] == 0).all() and (out[:, :, nw:] == 0).all()
    assert set(m.unique().tolist()) <= {0.0, 1.0}
    assert m[~valid].sum() == 0
    scale = 1000 / 1280
    assert abs(m.sum().item() - 200 * 300 * scale ** 2) / (200 * 300 * scale ** 2) < 0.02


def test_resize_pad_upscales_small_frames():
    out, _, valid = resize_pad(torch.rand(3, 240, 320), None, long_side=1024, patch=16)
    assert valid.shape == (768, 1024) and out.shape == (3, 768, 1024)


def test_unpad_resize_inverts_geometry():
    h, w = 720, 1280
    mask = torch.zeros(h, w)
    mask[200:400, 300:700] = 1
    _, m, _ = resize_pad(torch.rand(3, h, w), mask, 1024, 16)
    back = unpad_resize(m[None] * 20 - 10, resized_hw(h, w, 1024), (h, w))[0] > 0
    iou = (back & (mask > 0)).sum() / (back | (mask > 0)).sum()
    assert back.shape == (h, w) and iou > 0.98


def test_bucket_batches_are_homogeneous_and_complete():
    sizes = [(720, 1280), (1080, 1920), (240, 320), (480, 720)] * 9
    keys = [bucket_key(h, w, 1024, 16) for h, w in sizes]
    assert keys[0] == keys[1]  # 16:9 at any resolution lands in one bucket
    batches = make_bucket_batches(range(len(keys)), keys, 4, rng=np.random.default_rng(0))
    for b in batches:
        assert len({keys[i] for i in b}) == 1 and len(b) <= 4
    assert sorted(i for b in batches for i in b) == list(range(len(keys)))
