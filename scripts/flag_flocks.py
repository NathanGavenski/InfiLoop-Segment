"""Phase 1 flock flags: videos where zero-shot SAM 3 finds substantial foreground outside the
union mask, which usually means unannotated animals (birds in a flock, blurred groups).

Reads B2 predictions (scripts/b2_sam3_zero_shot.py) and the frame table. For each frame,
`outside` is the share of the frame's pixels that SAM marks foreground but the union mask
does not. A video is flagged when at least `--min-frame-share` of its scored frames have
`outside >= --min-outside`. Both thresholds are a starting proposal; inspect the flagged
list before relying on it.

    python scripts/flag_flocks.py --pred $WORK_DIR/outputs/b2/preds_test.parquet \\
        --frames $WORK_DIR/index/frames_test.parquet --out $WORK_DIR/flocks_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.sa_fari import decode  # noqa: E402


def outside_share(pred_rle, gt_rle, h: int, w: int) -> float:
    pred = decode(pred_rle, h, w).astype(bool)
    if not pred.any():
        return 0.0
    gt = decode(gt_rle, h, w).astype(bool)
    return float((pred & ~gt).sum() / (h * w))


def flag(frames: pd.DataFrame, pred: pd.DataFrame, min_outside: float,
         min_frame_share: float) -> tuple[list, pd.DataFrame]:
    df = frames.merge(pred[["video_id", "frame_idx", "pred_rle"]], on=["video_id", "frame_idx"])
    df["outside"] = [outside_share(p, g, h, w) for p, g, h, w in
                     zip(df.pred_rle, df.union_rle, df.height, df.width)]
    per_video = df.groupby("video_id").agg(
        video_name=("video_name", "first"), frames=("outside", "size"),
        frame_share=("outside", lambda x: float((x >= min_outside).mean())),
        median_outside=("outside", "median"))
    flagged = per_video[per_video.frame_share >= min_frame_share].index.tolist()
    return flagged, per_video


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", type=Path, nargs="+", required=True, help="B2 prediction file(s)")
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--min-outside", type=float, default=0.002,
                   help="share of frame pixels outside the union mask (default 0.2%%)")
    p.add_argument("--min-frame-share", type=float, default=0.25)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    pred = pd.concat([pd.read_parquet(x) for x in args.pred], ignore_index=True)
    flagged, per_video = flag(pd.read_parquet(args.frames), pred,
                              args.min_outside, args.min_frame_share)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "min_outside": args.min_outside, "min_frame_share": args.min_frame_share,
        "flagged_videos": [int(v) if isinstance(v, (np.integer, int)) else v for v in flagged],
        "flagged_names": per_video.loc[flagged, "video_name"].tolist()}, indent=1))
    per_video.to_csv(args.out.with_suffix(".csv"))
    print(f"flagged {len(flagged)} of {len(per_video)} videos -> {args.out}")


if __name__ == "__main__":
    main()
