import numpy as np
import pandas as pd
import pytest

import eval as ev
from data.sa_fari import encode

H, W = 100, 100


def box(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_iou_known_values():
    a, b = box(0, 10, 0, 10), box(0, 10, 5, 15)
    assert ev.iou(a, a) == 1.0
    assert ev.iou(a, b) == pytest.approx(50 / 150)
    assert ev.iou(a, box(50, 60, 50, 60)) == 0.0
    assert ev.iou(np.zeros((H, W), bool), np.zeros((H, W), bool)) == 1.0
    assert ev.iou(np.zeros((H, W), bool), a) == 0.0


def test_boundary_f():
    a = box(20, 60, 20, 60)
    assert ev.boundary_f(a, a) == 1.0
    # tolerance is ceil(0.008 * diag) = 2 px here; a 1 px shift is within it
    assert ev.boundary_f(box(21, 61, 20, 60), a) == 1.0
    assert ev.boundary_f(box(20, 60, 70, 95), a) < 0.2
    assert ev.boundary_f(np.zeros((H, W), bool), np.zeros((H, W), bool)) == 1.0
    assert ev.boundary_f(np.zeros((H, W), bool), a) == 0.0


def test_presence_pr_and_size_class():
    p, r = ev.presence_pr([1, 1, 0, 0], [1, 0, 1, 0])
    assert p == 0.5 and r == 0.5
    assert ev.size_class(31 ** 2) == "small" and ev.size_class(32 ** 2) == "medium"
    assert ev.size_class(96 ** 2) == "large"


def _toy():
    """Three frames: a large and a small animal, an empty frame, a lone small animal."""
    big, small1, small2 = box(10, 60, 10, 60), box(80, 85, 80, 85), box(5, 9, 90, 94)
    frames = pd.DataFrame({
        "video_id": [0, 0, 1], "frame_idx": [0, 1, 0], "height": H, "width": W,
        "union_rle": [encode(big | small1), None, encode(small2)],
    })
    inst = pd.DataFrame({
        "video_id": [0, 0, 1], "frame_idx": [0, 0, 0],
        "area": [float(big.sum()), float(small1.sum()), float(small2.sum())],
        "rle": [encode(big), encode(small1), encode(small2)],
        "species": ["deer", "mouse", "mouse"], "noun_phrase": ["deer", "mouse", "mouse"],
        "difficulty": [None, None, None],
    })
    # prediction: finds the deer only, a false positive on the empty frame, misses frame 3
    fp = box(0, 10, 0, 10)
    pred = pd.DataFrame({"video_id": [0, 0, 1], "frame_idx": [0, 1, 0],
                         "pred_rle": [encode(big), encode(fp), None],
                         "presence_score": np.nan, "loops": 1, "latency_ms": 5.0,
                         "is_night": [False, True, False]})
    return frames, inst, pred, (big, small1, small2, fp)


def test_frame_metrics_and_strata():
    frames, inst, pred, (big, small1, small2, fp) = _toy()
    df, inst_df = ev.frame_metrics(pred, frames, inst, workers=1)
    assert list(df.iou.round(6)) == [round(big.sum() / (big | small1).sum(), 6), 0.0, 0.0]
    assert list(df["size"]) == ["small", "empty", "small"]
    assert list(inst_df.coverage) == [1.0, 0.0, 0.0]
    res = ev.strata(df, inst_df, flock_videos={1}, presence_threshold=None, eps=0,
                    min_species_frames=2)
    o = res["overall"]
    inter = big.sum()
    union = (big | small1).sum() + fp.sum() + small2.sum()
    assert o["global_iou"] == pytest.approx(inter / union)
    assert o["presence_precision"] == 0.5 and o["presence_recall"] == 0.5
    assert o["mean_iou_positive"] == pytest.approx(df.iou[[0, 2]].mean())
    assert res["instance_coverage_by_size"]["small"]["instances"] == 2
    assert res["instance_coverage_by_size"]["medium"]["mean_coverage"] == 1.0  # 50x50 px
    assert res["day_night"]["night"]["frames"] == 1
    assert res["difficulty"].startswith("skipped")
    assert res["species"]["mouse"]["frames"] == 2
    assert "long tail (<2 frames)" in res["species"]  # deer appears in one frame
    assert res["flock"]["flagged"]["frames"] == 1


def test_presence_threshold_overrides_mask_area():
    frames, inst, pred, _ = _toy()
    pred["presence_score"] = [0.9, 0.1, 0.8]
    df, inst_df = ev.frame_metrics(pred, frames, inst, workers=1)
    o = ev.summarise(df, presence_threshold=0.5, eps=0)
    assert o["presence_precision"] == 1.0 and o["presence_recall"] == 1.0


def test_parallel_matches_serial():
    frames, inst, pred, _ = _toy()
    a, _ = ev.frame_metrics(pred, frames, inst, workers=1)
    b, _ = ev.frame_metrics(pred, frames, inst, workers=2)
    pd.testing.assert_frame_equal(a, b)
