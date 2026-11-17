import numpy as np
import pytest
from pycocotools import mask as mask_utils

from data.sa_fari import SAFari, decode, encode, sample_frames, split_by_location, to_rle, union_rle

H, W = 20, 30


def rle_of(mask: np.ndarray) -> dict:
    r = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": r["size"], "counts": r["counts"].decode("ascii")}  # as stored in JSON


def box(y0, y1, x0, x1):
    m = np.zeros((H, W), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def toy_dataset():
    a, b, c = box(0, 5, 0, 5), box(3, 8, 3, 8), box(10, 12, 20, 25)
    seg = lambda m: rle_of(m) if m is not None else None
    area = lambda m: float(m.sum()) if m is not None else None
    bbox = lambda m: mask_utils.toBbox(to_rle(rle_of(m), H, W)).tolist() if m is not None else None
    ann = lambda i, cat, ms: {"id": i, "video_id": 7, "category_id": cat, "noun_phrase": f"c{cat}",
                              "iscrowd": 0, "height": H, "width": W,
                              "segmentations": [seg(m) for m in ms],
                              "areas": [area(m) for m in ms], "bboxes": [bbox(m) for m in ms]}
    return {
        "info": {}, "video_np_pairs": [],
        "videos": [{"id": 7, "video_name": "v7", "height": H, "width": W, "length": 3,
                    "file_names": ["v7/0.jpg", "v7/1.jpg", "v7/2.jpg"], "location_id": "L1"}],
        "categories": [{"id": 1, "name": "fox", "Species": "vulpes vulpes", "Class": "Mammalia"},
                       {"id": 2, "name": "bird", "Species": float("nan")}],
        # frame 0: overlapping a and b (two species); frame 1: c only; frame 2: empty
        "annotations": [ann(1, 1, [a, c, None]), ann(2, 2, [b, None, None])],
    }, (a, b, c)


def test_union_of_overlapping_masklets_across_species():
    data, (a, b, c) = toy_dataset()
    ds = SAFari(data)
    np.testing.assert_array_equal(ds.union_mask(7, 0), (a | b))
    np.testing.assert_array_equal(ds.union_mask(7, 1), c)
    assert ds.union_mask(7, 0).sum() == 25 + 25 - 4  # overlap counted once


def test_empty_frame_gives_zero_mask_and_no_presence():
    data, _ = toy_dataset()
    ds = SAFari(data)
    assert ds.union_rle(7, 2) is None
    assert ds.union_mask(7, 2).shape == (H, W) and ds.union_mask(7, 2).sum() == 0
    assert [ds.presence(7, t) for t in range(3)] == [1, 1, 0]


def test_tables_match_union_and_index_rows():
    data, (a, b, c) = toy_dataset()
    frames, inst = SAFari(data).tables()
    assert list(frames.presence) == [1, 1, 0]
    assert list(frames.n_instances) == [2, 1, 0]
    np.testing.assert_array_equal(decode(frames.union_rle[0], H, W), a | b)
    assert frames.union_area[0] == (a | b).sum()
    assert frames.union_rle[2] is None
    assert len(inst) == 3  # one row per instance per frame
    row = inst[(inst.frame_idx == 0) & (inst.category_id == 1)].iloc[0]
    assert row.area == 25 and row.species == "vulpes vulpes" and row["class"] == "Mammalia"
    assert (row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h) == (0, 0, 5, 5)
    assert row.location_id == "L1"
    assert inst[inst.category_id == 2].species.isna().all()  # NaN taxonomy becomes null


def test_uncompressed_rle_and_size_mismatch():
    m = box(2, 6, 4, 9)
    # build an uncompressed RLE by hand: column-major run lengths
    flat = m.flatten(order="F")
    runs, cur, n = [], 0, 0
    for v in flat:
        if v != cur:
            runs.append(n); cur, n = v, 0
        n += 1
    runs.append(n)
    unc = {"size": [H, W], "counts": runs}
    np.testing.assert_array_equal(decode(to_rle(unc, H, W), H, W), m)
    with pytest.raises(ValueError):
        union_rle([to_rle(rle_of(m), H, W)], H + 1, W)


def test_encode_decode_roundtrip():
    m = box(1, 4, 2, 7) | box(10, 15, 20, 29)
    assert encode(np.zeros((H, W), np.uint8)) is None
    np.testing.assert_array_equal(decode(encode(m), H, W), m)


def test_location_split_is_deterministic_and_disjoint():
    locs = [f"R{i:03d}" for i in range(100)] * 3
    val = split_by_location(locs, 0.05, seed=0)
    assert len(val) == 5 and val == split_by_location(locs, 0.05, seed=0)
    assert val != split_by_location(locs, 0.05, seed=1)


def test_sample_frames_stratified():
    import pandas as pd
    frames = pd.DataFrame({"video_id": [0] * 10 + [1] * 4,
                           "presence": [1] * 6 + [0] * 4 + [1] * 4})
    idx = sample_frames(frames, per_video=4, empty_share=0.5, rng=np.random.default_rng(0))
    v0 = frames.loc[idx][frames.loc[idx].video_id == 0]
    assert len(v0) == 4 and v0.presence.sum() == 2
    v1 = frames.loc[idx][frames.loc[idx].video_id == 1]
    assert len(v1) == 4 and v1.presence.all()  # no empties, positives fill in
    assert len(set(idx)) == len(idx)


def test_missing_locations_never_go_to_validation():
    import pandas as pd
    from data.sa_fari import assign_split
    locs = [f"R{i}" for i in range(40)] + [float("nan"), None]
    val = split_by_location(locs, 0.5, seed=0)
    assert "nan" not in val and "None" not in val and len(val) == 20
    frames = pd.DataFrame({"location_id": [next(iter(val)), float("nan"), None, "zzz"]})
    assert list(assign_split(frames, val)) == ["val", "train", "train", "train"]
