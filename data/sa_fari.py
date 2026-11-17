"""SA-FARI parsing: per-frame union masks, presence labels, frame table and instance index.

SA-FARI follows the SA-Co/VEval format, which is YTVIS-like. Each annotation is a masklet:
one segmentation per video frame (an RLE dict, or None when the animal is absent from that
frame). We merge the masklets of every species in a frame into one binary union mask, and
the presence label is 1 when that union is non-empty.

Two Parquet tables are written per annotation file, and every later stage reads them:

* ``frames_<split>.parquet``: one row per frame, with the union mask as a compressed RLE
  string, presence, location and the boxes and areas of the frame's instances (used by
  small-animal crops). The training loader reads only this table.
* ``instances_<split>.parquet``: one row per instance per frame with area, bbox, species,
  taxonomy, video ID, location ID and difficulty tag. Strata are computed from this table.

CLI::

    python -m data.sa_fari --ann $SAFARI_ROOT/annotation/sa_fari_train_ext.json \\
        --split train --out $WORK_DIR/index --val-frac 0.05 --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

TAXONOMY_KEYS = ("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")
# The SA-FARI card documents no difficulty field. Any of these keys is picked up if a later
# release adds one; otherwise the column is null and the difficulty stratum is skipped.
DIFFICULTY_KEYS = ("difficulty", "difficulty_tag", "difficulty_tags", "is_difficult", "hard")


# --------------------------------------------------------------------------- RLE helpers

def to_rle(seg: dict | list | None, height: int, width: int) -> dict | None:
    """Normalise one segmentation to a compressed RLE dict (counts as bytes)."""
    if seg is None:
        return None
    if isinstance(seg, list):  # polygon(s)
        rles = mask_utils.frPyObjects(seg, height, width)
        return mask_utils.merge(rles)
    if isinstance(seg.get("counts"), list):  # uncompressed RLE
        return mask_utils.frPyObjects(seg, *seg["size"])
    counts = seg["counts"]
    return {"size": list(seg["size"]),
            "counts": counts.encode("ascii") if isinstance(counts, str) else counts}


def union_rle(rles: Iterable[dict | None], height: int, width: int) -> dict | None:
    """Union of RLEs, or None when every input is None (empty frame)."""
    rles = [r for r in rles if r is not None]
    if not rles:
        return None
    for r in rles:
        if tuple(r["size"]) != (height, width):
            raise ValueError(f"RLE size {r['size']} does not match frame {height}x{width}")
    return rles[0] if len(rles) == 1 else mask_utils.merge(rles, intersect=False)


def decode(rle: dict | str | None, height: int, width: int) -> np.ndarray:
    """Decode an RLE dict, or the counts string stored in Parquet, to a uint8 HxW mask."""
    if rle is None or (isinstance(rle, float) and math.isnan(rle)):
        return np.zeros((height, width), dtype=np.uint8)
    if isinstance(rle, (str, bytes)):
        rle = {"size": [height, width],
               "counts": rle.encode("ascii") if isinstance(rle, str) else rle}
    return mask_utils.decode(rle).astype(np.uint8)


def encode(mask: np.ndarray) -> str | None:
    """Encode a binary HxW mask to the compressed counts string, or None if empty."""
    if not mask.any():
        return None
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


# --------------------------------------------------------------------------- dataset

class SAFari:
    """In-memory view of one SA-FARI annotation file."""

    def __init__(self, ann: str | Path | dict):
        data = ann if isinstance(ann, dict) else json.loads(Path(ann).read_text())
        self.info = data.get("info", {})
        self.videos: dict[Any, dict] = {v["id"]: v for v in data["videos"]}
        self.categories: dict[int, dict] = {c["id"]: c for c in data["categories"]}
        self.anns_by_video: dict[Any, list[dict]] = defaultdict(list)
        for a in data["annotations"]:
            if a["video_id"] not in self.videos:
                raise KeyError(f"annotation {a['id']} points at unknown video {a['video_id']}")
            if len(a["segmentations"]) != self.videos[a["video_id"]]["length"]:
                raise ValueError(f"annotation {a['id']}: {len(a['segmentations'])} masks for a "
                                 f"{self.videos[a['video_id']]['length']}-frame video")
            self.anns_by_video[a["video_id"]].append(a)

    def frame_rles(self, video_id, frame_idx: int) -> list[tuple[dict, dict]]:
        """(annotation, rle) for every masklet present in this frame."""
        v = self.videos[video_id]
        out = []
        for a in self.anns_by_video.get(video_id, []):
            rle = to_rle(a["segmentations"][frame_idx], v["height"], v["width"])
            if rle is not None:
                out.append((a, rle))
        return out

    def union_rle(self, video_id, frame_idx: int) -> dict | None:
        v = self.videos[video_id]
        return union_rle((r for _, r in self.frame_rles(video_id, frame_idx)),
                         v["height"], v["width"])

    def union_mask(self, video_id, frame_idx: int) -> np.ndarray:
        v = self.videos[video_id]
        return decode(self.union_rle(video_id, frame_idx), v["height"], v["width"])

    def presence(self, video_id, frame_idx: int) -> int:
        return int(any(a["segmentations"][frame_idx] is not None
                       for a in self.anns_by_video.get(video_id, [])))

    @staticmethod
    def _difficulty(*dicts: dict) -> str | None:
        for d in dicts:
            for k in DIFFICULTY_KEYS:
                if d.get(k) is not None:
                    return str(d[k])
        return None

    def tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build the frame table and the instance index."""
        frames, instances = [], []
        for vid, v in self.videos.items():
            h, w = v["height"], v["width"]
            anns = self.anns_by_video.get(vid, [])
            for t, file_name in enumerate(v["file_names"]):
                rles, boxes, areas = [], [], []
                for a in anns:
                    seg = a["segmentations"][t]
                    if seg is None:
                        continue
                    rle = to_rle(seg, h, w)
                    area = a["areas"][t] if a.get("areas") and a["areas"][t] is not None \
                        else float(mask_utils.area(rle))
                    bbox = a["bboxes"][t] if a.get("bboxes") and a["bboxes"][t] is not None \
                        else mask_utils.toBbox(rle).tolist()
                    rles.append(rle)
                    boxes.append([float(x) for x in bbox])
                    areas.append(float(area))
                    cat = self.categories.get(a["category_id"], {})
                    instances.append({
                        "video_id": vid, "video_name": v["video_name"],
                        "location_id": _clean(v.get("location_id")), "frame_idx": t,
                        "file_name": file_name, "annotation_id": a["id"],
                        "category_id": a["category_id"], "noun_phrase": a.get("noun_phrase"),
                        **{k.lower(): _clean(cat.get(k)) for k in TAXONOMY_KEYS},
                        "area": float(area),
                        "bbox_x": bbox[0], "bbox_y": bbox[1], "bbox_w": bbox[2], "bbox_h": bbox[3],
                        "height": h, "width": w, "iscrowd": a.get("iscrowd", 0),
                        "difficulty": self._difficulty(a, v),
                    })
                u = union_rle(rles, h, w)
                frames.append({
                    "video_id": vid, "video_name": v["video_name"],
                    "location_id": _clean(v.get("location_id")), "frame_idx": t,
                    "file_name": file_name, "height": h, "width": w,
                    "presence": int(u is not None), "n_instances": len(rles),
                    "union_area": float(mask_utils.area(u)) if u is not None else 0.0,
                    "union_rle": u["counts"].decode("ascii") if u is not None else None,
                    "inst_bboxes": boxes, "inst_areas": areas,
                })
        frame_df = pd.DataFrame(frames)
        inst_df = pd.DataFrame(instances, columns=_INSTANCE_COLUMNS)
        return frame_df, inst_df


_INSTANCE_COLUMNS = [
    "video_id", "video_name", "location_id", "frame_idx", "file_name", "annotation_id",
    "category_id", "noun_phrase", *[k.lower() for k in TAXONOMY_KEYS], "area",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h", "height", "width", "iscrowd", "difficulty",
]


def _clean(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


# --------------------------------------------------------------------------- splits

def split_by_location(location_ids: Iterable[str], val_frac: float = 0.05,
                      seed: int = 0) -> set[str]:
    """Pick about `val_frac` of the locations for validation, deterministically.

    Videos without a location ID (NaN in the _ext files) are never candidates, so they stay
    in training; we cannot tell which camera they come from.
    """
    locs = sorted({str(x) for x in location_ids if _clean(x) is not None})
    if not locs:
        raise ValueError("no location IDs; use the _ext annotation files")
    n_val = max(1, round(val_frac * len(locs)))
    return set(random.Random(seed).sample(locs, n_val))


def write_splits(path: Path, val_locs: set[str], val_frac: float, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"val_frac": val_frac, "seed": seed,
                                "val_locations": sorted(val_locs)}, indent=1) + "\n")
    print(f"Wrote {len(val_locs)} validation locations to {path}")


def assign_split(frames: pd.DataFrame, val_locations: set[str], base: str = "train") -> pd.Series:
    loc = frames["location_id"]
    return np.where(loc.notna() & loc.astype(str).isin(val_locations), "val", base)


# --------------------------------------------------------------------------- sampling

def sample_frames(frames: pd.DataFrame, per_video: int, empty_share: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Draw `per_video` frame row-indices per video, stratified by presence.

    About `empty_share` of each video's draw comes from frames without animals when the video
    has any; the rest are positive frames. If one stratum runs short the other fills in.
    """
    out = []
    for _, g in frames.groupby("video_id", sort=False):
        idx = g.index.to_numpy()
        pres = g["presence"].to_numpy().astype(bool)
        pos, neg = idx[pres], idx[~pres]
        k = min(per_video, len(idx))
        n_neg = min(len(neg), int(round(k * empty_share)))
        n_pos = min(len(pos), k - n_neg)
        n_neg = min(len(neg), k - n_pos)
        if n_pos:
            out.append(rng.choice(pos, n_pos, replace=False))
        if n_neg:
            out.append(rng.choice(neg, n_neg, replace=False))
    return np.concatenate(out) if out else np.array([], dtype=int)


# --------------------------------------------------------------------------- day/night

def is_night(image: np.ndarray, tol: float = 2.0) -> bool:
    """Infrared night frames have equal colour channels. JPEG noise breaks exact equality,
    so a frame is night when the mean absolute channel spread is below `tol` grey levels."""
    if image.ndim == 2 or image.shape[-1] == 1:
        return True
    img = image.astype(np.int16)
    spread = np.maximum(np.abs(img[..., 0] - img[..., 1]), np.abs(img[..., 1] - img[..., 2]))
    return float(spread.mean()) < tol


# --------------------------------------------------------------------------- CLI

def main() -> None:
    p = argparse.ArgumentParser(description="Build frame table and instance index")
    p.add_argument("--ann", type=Path, required=True, help="SA-FARI annotation JSON (_ext)")
    p.add_argument("--split", required=True, choices=["train", "test"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="Share of training locations held out as validation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--splits-file", type=Path, default=Path("splits/sa_fari_val_locations.json"))
    p.add_argument("--splits-only", action="store_true",
                   help="Only write the validation-location file (light parse, no tables)")
    args = p.parse_args()

    if args.splits_only:
        with args.ann.open() as f:
            data = json.load(f, object_hook=lambda o: None if "counts" in o and "size" in o else o)
        val_locs = split_by_location((v.get("location_id") for v in data["videos"]),
                                     args.val_frac, args.seed)
        write_splits(args.splits_file, val_locs, args.val_frac, args.seed)
        return

    ds = SAFari(args.ann)
    frames, inst = ds.tables()
    if args.split == "train":
        if args.splits_file.exists():
            val_locs = set(json.loads(args.splits_file.read_text())["val_locations"])
            print(f"Using existing validation locations from {args.splits_file}")
        else:
            val_locs = split_by_location(frames["location_id"], args.val_frac, args.seed)
            write_splits(args.splits_file, val_locs, args.val_frac, args.seed)
        frames["split"] = assign_split(frames, val_locs)
        inst["split"] = assign_split(inst, val_locs)
    else:
        frames["split"] = "test"
        inst["split"] = "test"

    args.out.mkdir(parents=True, exist_ok=True)
    frames.to_parquet(args.out / f"frames_{args.split}.parquet", index=False)
    inst.to_parquet(args.out / f"instances_{args.split}.parquet", index=False)
    for s, g in frames.groupby("split"):
        print(f"{s:5s} videos={g.video_id.nunique():6d} locations={g.location_id.nunique():4d} "
              f"frames={len(g):8d} empty={1 - g.presence.mean():.3f}")
    print(f"instances={len(inst)} difficulty tagged={inst.difficulty.notna().sum()}")


if __name__ == "__main__":
    main()
