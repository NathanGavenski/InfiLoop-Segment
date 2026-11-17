"""Baseline B2: zero-shot SAM 3.1 prompted with "animal", masks merged per frame.

Runs the SAM 3 image predictor on every frame of a split, keeps instance masks whose score
is at least `sam3.score_threshold`, takes their union and writes the standard prediction
file (see eval.py), so B2 is scored by exactly the same code as B0 and B1:

    python scripts/b2_sam3_zero_shot.py --config configs/b2_sam3.yaml --split test \\
        --out $WORK_DIR/outputs/b2/preds_test.parquet [--shard 0 --num-shards 8]
    python eval.py score --pred $WORK_DIR/outputs/b2/preds_test.parquet --index-dir $WORK_DIR/index ...

With `--oracle` (or sam3.oracle_species_prompts: true) each video is prompted with its
ground-truth noun phrases instead. That leaks labels, so report it apart from B2.

Frames are processed independently with the image model, which matches the per-frame task.
SAM 3 weights are under the SAM License; outputs on SA-FARI are non-commercial.

The sam3 calls live in `Sam3Runner` only. They were checked against the sam3 source at the
commit pinned in pyproject.toml (build_sam3_image_model, Sam3Processor), but not yet run
with the real weights.

SAM 3 vs 3.1: the per-frame image model loads `facebook/sam3` (sam3.pt) by default. SAM 3.1
ships as a video multiplex checkpoint (`facebook/sam3.1`, sam3.1_multiplex.pt). With
`sam3.version: sam3.1` its detector weights are loaded into the image model; sam3 prints any
missing keys while loading, so check the log before trusting that variant.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.sa_fari import encode, is_night  # noqa: E402
from eval import PRED_COLUMNS  # noqa: E402
from utils.config import load_config  # noqa: E402


class Sam3Runner:
    """Thin adapter over the sam3 image API: image + text prompts -> (masks, scores)."""

    def __init__(self, checkpoint: str | None, score_threshold: float, version: str = "sam3"):
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if checkpoint is None and version == "sam3.1":
            checkpoint = download_ckpt_from_hf(version="sam3.1")
        self.model = build_sam3_image_model(checkpoint_path=checkpoint, device=device)
        # The processor already drops instances scoring below the threshold.
        self.processor = Sam3Processor(self.model, device=device,
                                       confidence_threshold=score_threshold)
        self.threshold = score_threshold

    @torch.inference_mode()
    def __call__(self, image: Image.Image, prompts: list[str]) -> tuple[np.ndarray, float]:
        """Union of kept masks over all prompts (HxW bool) and the best instance score."""
        state = self.processor.set_image(image)
        union = np.zeros((image.height, image.width), dtype=bool)
        best = 0.0
        for prompt in prompts:
            out = self.processor.set_text_prompt(state=state, prompt=prompt)
            masks, scores = out["masks"], out["scores"]
            if len(scores) == 0:
                continue
            scores = scores.float().cpu().numpy().reshape(-1)
            masks = masks.cpu().numpy().reshape(len(scores), *masks.shape[-2:]) > 0.5
            keep = scores >= self.threshold
            if keep.any():
                union |= masks[keep].any(0)
            best = max(best, float(scores.max()))
        return union, best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/b2_sam3.yaml"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--oracle", action="store_true", help="prompt with GT noun phrases (leaks labels)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--frame-stride", type=int, default=1, help="every n-th frame (cost control)")
    p.add_argument("--limit-videos", type=int)
    p.add_argument("--set", nargs="*", default=[])
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    s3, d = cfg["sam3"], cfg["data"]
    oracle = args.oracle or s3.get("oracle_species_prompts", False)
    split_file = "train" if args.split in ("train", "val") else "test"
    frames = pd.read_parquet(Path(d["index_dir"]) / f"frames_{split_file}.parquet")
    frames = frames[frames["split"] == args.split]
    videos = sorted(frames.video_id.unique())[args.shard::args.num_shards]
    if args.limit_videos:
        videos = videos[:args.limit_videos]
    frames = frames[frames.video_id.isin(videos) & (frames.frame_idx % args.frame_stride == 0)]
    prompts_by_video: dict = {}
    if oracle:
        inst = pd.read_parquet(Path(d["index_dir"]) / f"instances_{split_file}.parquet",
                               columns=["video_id", "noun_phrase"])
        prompts_by_video = inst.groupby("video_id").noun_phrase.agg(lambda x: sorted(set(x))).to_dict()

    out = args.out if args.num_shards == 1 else args.out.with_name(
        f"{args.out.stem}.shard{args.shard:03d}of{args.num_shards:03d}{args.out.suffix}")
    done = set()
    rows = []
    if out.exists():  # resume
        prev = pd.read_parquet(out)
        rows = prev.to_dict("records")
        done = set(zip(prev.video_id, prev.frame_idx))
    print(f"{len(frames)} frames in {len(videos)} videos, {len(done)} already done -> {out}")

    runner = Sam3Runner(s3.get("checkpoint"), s3.get("score_threshold", 0.5),
                        s3.get("version", "sam3"))
    root = Path(d[f"image_root_{split_file}"])
    cuda = torch.cuda.is_available()
    for n, row in enumerate(frames.itertuples(index=False)):
        if (row.video_id, row.frame_idx) in done:
            continue
        image = Image.open(root / row.file_name).convert("RGB")
        prompts = prompts_by_video.get(row.video_id, []) if oracle else [s3.get("prompt", "animal")]
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        mask, score = runner(image, prompts) if prompts else \
            (np.zeros((row.height, row.width), bool), 0.0)
        if cuda:
            torch.cuda.synchronize()
        rows.append({"video_id": row.video_id, "frame_idx": int(row.frame_idx),
                     "pred_rle": encode(mask), "presence_score": score, "loops": 1,
                     "latency_ms": (time.perf_counter() - t0) * 1000,
                     "is_night": is_night(np.asarray(image))})
        if len(rows) % 500 == 0:
            pd.DataFrame(rows, columns=PRED_COLUMNS).to_parquet(out, index=False)
            print(f"  {n + 1}/{len(frames)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=PRED_COLUMNS).to_parquet(out, index=False)
    print(f"wrote {len(rows)} predictions to {out}"
          + (" (ORACLE prompts: report separately)" if oracle else ""))


if __name__ == "__main__":
    main()
