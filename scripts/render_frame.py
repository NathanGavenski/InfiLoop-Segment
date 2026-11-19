"""Phase 0 check: render one SA-FARI frame with its union mask overlaid, to a PNG.

    python scripts/render_frame.py --frames $WORK_DIR/index/frames_test.parquet \\
        --image-root $SAFARI_ROOT/sa_fari_test --video sa_fari_000000 --out frame.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.sa_fari import decode  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--video", required=True, help="video_name, e.g. sa_fari_000000")
    p.add_argument("--frame-idx", type=int, help="default: first frame with an animal")
    p.add_argument("--out", type=Path, default=Path("frame_overlay.png"))
    args = p.parse_args()

    df = pd.read_parquet(args.frames)
    rows = df[df.video_name == args.video]
    row = rows[rows.frame_idx == args.frame_idx].iloc[0] if args.frame_idx is not None \
        else rows[rows.presence == 1].iloc[0]
    img = np.asarray(Image.open(args.image_root / row.file_name).convert("RGB")).copy()
    mask = decode(row.union_rle, row.height, row.width).astype(bool)
    img[mask] = (0.5 * img[mask] + 0.5 * np.array([255, 0, 160])).astype(np.uint8)
    Image.fromarray(img).save(args.out)
    print(f"{row.file_name} presence={row.presence} instances={row.n_instances} "
          f"union_area={row.union_area:.0f} -> {args.out}")


if __name__ == "__main__":
    main()
