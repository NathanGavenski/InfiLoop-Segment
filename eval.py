"""Evaluation: prediction export, metrics and strata (PLAN.md Phase 5).

Every method (B0, B1, B2 and later the looped model) writes the same prediction file, one row
per frame at original resolution::

    video_id, frame_idx, pred_rle (compressed counts or null), presence_score (optional),
    loops, latency_ms, is_night

and `score` compares it with the frame table and instance index. Metrics:

* mean per-frame foreground IoU (a frame where both masks are empty scores 1), also over
  positive frames only
* global IoU, summing intersections and unions over all pixels
* boundary F-score (DAVIS style, tolerance 0.8% of the image diagonal)
* presence precision and recall: a frame is predicted positive when presence_score >= the
  threshold if the method gives a score, else when the mask has more than `eps` pixels
* mean loops per frame and latency per frame
* per-instance coverage (share of an instance's pixels predicted foreground), because one
  binary mask cannot be matched to instances

Strata: instance size (COCO 32^2 / 96^2 in original pixels; a frame takes the class of its
smallest instance), day/night, difficulty (skipped when the annotations carry no tags),
species (long tail grouped) and flock flag.

    python eval.py predict --config configs/b1_dinov3.yaml --ckpt run/best.ckpt --out preds.parquet
    python eval.py score --pred preds.parquet --index-dir $WORK_DIR/index --out metrics.json
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from data.sa_fari import decode

SMALL, MEDIUM = 32 ** 2, 96 ** 2
PRED_COLUMNS = ["video_id", "frame_idx", "pred_rle", "presence_score", "loops", "latency_ms",
                "is_night"]


# --------------------------------------------------------------------------- per-frame metrics

def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Foreground IoU of two binary masks; 1.0 when both are empty."""
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def _boundary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), bool), border_value=0)
    return mask & ~eroded


def boundary_f(pred: np.ndarray, gt: np.ndarray, bound_th: float = 0.008) -> float:
    """DAVIS boundary F-measure. Tolerance is bound_th * image diagonal (in pixels) when
    bound_th < 1, else bound_th pixels. Both empty -> 1, one empty -> 0."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    r = bound_th if bound_th >= 1 else np.ceil(bound_th * np.hypot(*gt.shape))
    pb, gb = _boundary(pred), _boundary(gt)
    if not pb.any() and not gb.any():
        return 1.0
    if not pb.any() or not gb.any():
        return 0.0
    d_to_gt = ndimage.distance_transform_edt(~gb)
    d_to_pred = ndimage.distance_transform_edt(~pb)
    precision = float((d_to_gt[pb] <= r).mean())
    recall = float((d_to_pred[gb] <= r).mean())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def size_class(area: float) -> str:
    return "small" if area < SMALL else "medium" if area < MEDIUM else "large"


def presence_pr(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    tp = (pred & gt).sum()
    precision = tp / pred.sum() if pred.sum() else float("nan")
    recall = tp / gt.sum() if gt.sum() else float("nan")
    return float(precision), float(recall)


# --------------------------------------------------------------------------- scoring

def _score_frame(item, bound_th: float):
    """Worker: decode one frame's masks and compute pixel counts, IoU, boundary F and the
    coverage of each instance."""
    gt_rle, pred_rle, h, w, inst_rles = item
    gt = decode(gt_rle, h, w).astype(bool)
    pred = decode(pred_rle, h, w).astype(bool)
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    cover = []
    for r in inst_rles:
        m = decode(r, h, w).astype(bool)
        cover.append(float(np.logical_and(m, pred).sum() / max(m.sum(), 1)))
    return {"inter": inter, "union": union, "pred_area": int(pred.sum()),
            "gt_area": int(gt.sum()), "iou": 1.0 if union == 0 else inter / union,
            "boundary_f": boundary_f(pred, gt, bound_th), "inst_coverage": cover}


def frame_metrics(pred: pd.DataFrame, frames: pd.DataFrame, instances: pd.DataFrame,
                  workers: int = 8, bound_th: float = 0.008) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join predictions to ground truth and compute per-frame and per-instance metrics."""
    key = ["video_id", "frame_idx"]
    df = frames.merge(pred, on=key, how="inner", validate="one_to_one")
    missing = len(frames) - len(df)
    if missing:
        print(f"WARNING: {missing} ground-truth frames have no prediction and are skipped")
    inst = instances.set_index(key).sort_index()
    inst_groups = {k: g for k, g in inst.groupby(level=[0, 1])} if len(inst) else {}
    items = []
    for row in df.itertuples(index=False):
        g = inst_groups.get((row.video_id, row.frame_idx))
        items.append((row.union_rle, row.pred_rle, row.height, row.width,
                      [] if g is None else list(g["rle"])))
    fn = partial(_score_frame, bound_th=bound_th)
    if workers > 1:
        with Pool(workers) as pool:
            res = pool.map(fn, items, chunksize=64)
    else:
        res = [fn(i) for i in items]
    res_df = pd.DataFrame(res)
    df = pd.concat([df.reset_index(drop=True), res_df.drop(columns="inst_coverage")], axis=1)

    inst_rows = []
    for row, r in zip(df.itertuples(index=False), res):
        g = inst_groups.get((row.video_id, row.frame_idx))
        if g is None:
            continue
        for (_, irow), cov in zip(g.iterrows(), r["inst_coverage"]):
            inst_rows.append({"video_id": row.video_id, "frame_idx": row.frame_idx,
                              "area": irow["area"], "size": size_class(irow["area"]),
                              "species": irow.get("species") or irow.get("noun_phrase"),
                              "difficulty": irow.get("difficulty"), "coverage": cov})
    inst_df = pd.DataFrame(inst_rows, columns=["video_id", "frame_idx", "area", "size",
                                               "species", "difficulty", "coverage"])
    df["size"] = _frame_size_class(df, inst_df)
    return df, inst_df


def _frame_size_class(df: pd.DataFrame, inst_df: pd.DataFrame) -> pd.Series:
    if len(inst_df) == 0:
        return pd.Series(["empty"] * len(df), index=df.index)
    smallest = inst_df.groupby(["video_id", "frame_idx"])["area"].min()
    idx = pd.MultiIndex.from_frame(df[["video_id", "frame_idx"]])
    return pd.Series([size_class(a) if not np.isnan(a) else "empty"
                      for a in smallest.reindex(idx).to_numpy()], index=df.index)


def summarise(df: pd.DataFrame, presence_threshold: float | None, eps: int) -> dict:
    if len(df) == 0:
        return {"frames": 0}
    gt_pos = df["gt_area"].to_numpy() > 0
    if presence_threshold is not None and df["presence_score"].notna().all():
        pred_pos = df["presence_score"].to_numpy() >= presence_threshold
    else:
        pred_pos = df["pred_area"].to_numpy() > eps
    p, r = presence_pr(pred_pos, gt_pos)
    union = df["union"].sum()
    out = {
        "frames": int(len(df)),
        "mean_iou": float(df["iou"].mean()),
        "mean_iou_positive": float(df.loc[gt_pos, "iou"].mean()) if gt_pos.any() else float("nan"),
        "global_iou": float(df["inter"].sum() / union) if union else 1.0,
        "boundary_f": float(df["boundary_f"].mean()),
        "presence_precision": p, "presence_recall": r,
        "mean_loops": float(df["loops"].mean()) if "loops" in df else float("nan"),
        "latency_ms": float(df["latency_ms"].mean()) if "latency_ms" in df else float("nan"),
    }
    return out


def strata(df: pd.DataFrame, inst_df: pd.DataFrame, flock_videos: set | None,
           presence_threshold: float | None, eps: int, min_species_frames: int = 50) -> dict:
    """Metrics overall and per stratum (PLAN.md Phase 5 table)."""
    s = lambda d: summarise(d, presence_threshold, eps)
    out = {"overall": s(df)}
    out["size"] = {k: s(g) for k, g in df.groupby("size")}
    out["instance_coverage_by_size"] = {
        k: {"instances": int(len(g)), "mean_coverage": float(g.coverage.mean())}
        for k, g in inst_df.groupby("size")}
    if "is_night" in df and df["is_night"].notna().any():
        out["day_night"] = {("night" if k else "day"): s(g)
                            for k, g in df.groupby(df["is_night"].astype(bool))}
    tagged = inst_df.dropna(subset=["difficulty"])
    if len(tagged):
        diff = tagged.groupby(["video_id", "frame_idx"])["difficulty"].agg(
            lambda x: ",".join(sorted(set(map(str, x)))))
        keyed = df.set_index(["video_id", "frame_idx"])
        keyed["difficulty"] = diff.reindex(keyed.index).fillna("untagged")
        out["difficulty"] = {k: s(g) for k, g in keyed.reset_index().groupby("difficulty")}
    else:
        out["difficulty"] = "skipped: no difficulty tags in the annotations"
    if len(inst_df):
        sp = inst_df[["video_id", "frame_idx", "species"]].drop_duplicates()
        counts = sp.species.value_counts()
        sp["group"] = sp.species.where(sp.species.map(counts) >= min_species_frames,
                                       f"long tail (<{min_species_frames} frames)")
        joined = df.merge(sp[["video_id", "frame_idx", "group"]], on=["video_id", "frame_idx"])
        out["species"] = {k: s(g) for k, g in joined.groupby("group")}
    if flock_videos is not None:
        fl = df["video_id"].isin(flock_videos)
        out["flock"] = {"flagged": s(df[fl]), "not_flagged": s(df[~fl])}
    if "loops" in df and df["loops"].nunique() > 1:
        out["by_loops"] = {int(k): s(g) for k, g in df.groupby("loops")}
    return out


# --------------------------------------------------------------------------- prediction

def night_from_tensor(image, valid, tol: float = 2.0) -> bool:
    """is_night on a 3xHxW [0, 1] tensor, ignoring padding (see data.sa_fari.is_night)."""
    img = image[:, valid] * 255
    spread = np.maximum((img[0] - img[1]).abs().numpy(), (img[1] - img[2]).abs().numpy())
    return bool(spread.mean() < tol)


def predict(cfg: dict, ckpt: Path, split: str, out: Path, weights: str = "ema",
            limit: int | None = None) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader

    from data.buckets import make_bucket_batches, unpad_resize
    from data.dataset import FrameDataset, collate
    from data.sa_fari import encode
    from models.segmenter import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    state = torch.load(ckpt, map_location="cpu")
    res = model.load_state_dict(state[weights], strict=False)
    bad = [k for k in res.missing_keys if not k.startswith("encoder.")] + res.unexpected_keys
    if bad:
        raise RuntimeError(f"checkpoint does not match the model: {bad[:10]}")
    model.eval()

    d = cfg["data"]
    split_file = "train" if split in ("train", "val") else "test"
    frames = pd.read_parquet(Path(d["index_dir"]) / f"frames_{split_file}.parquet")
    frames = frames[frames["split"] == split].reset_index(drop=True)
    if limit:
        frames = frames[frames.video_id.isin(frames.video_id.unique()[:limit])].reset_index(drop=True)
    ds = FrameDataset(frames, d[f"image_root_{split_file}"], d["long_side"],
                      model.encoder.patch_size, d.get("square", False))
    bs = cfg.get("eval", {}).get("batch_size", 1)
    batches = make_bucket_batches(range(len(ds)), ds.keys, bs)
    loader = DataLoader(ds, batch_sampler=batches, collate_fn=collate,
                        num_workers=cfg.get("eval", {}).get("num_workers", 4))
    rows = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_m = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000 / len(images)
        logits = out_m["logits"][:, 0].float().cpu()
        for j, i in enumerate(batch["index"].tolist()):
            row = frames.iloc[i]
            full = unpad_resize(logits[j], tuple(batch["resized_hw"][j].tolist()),
                                (int(row.height), int(row.width)))
            rows.append({"video_id": row.video_id, "frame_idx": int(row.frame_idx),
                         "pred_rle": encode((full > 0).numpy()),
                         "presence_score": float(torch.sigmoid(full.max())),
                         "loops": int(out_m.get("loops", 1)), "latency_ms": ms,
                         "is_night": night_from_tensor(batch["image"][j], batch["valid"][j])})
    pred = pd.DataFrame(rows, columns=PRED_COLUMNS)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(out, index=False)
    print(f"wrote {len(pred)} predictions to {out}")
    return pred


# --------------------------------------------------------------------------- CLI

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("predict")
    pp.add_argument("--config", type=Path, required=True)
    pp.add_argument("--ckpt", type=Path, required=True)
    pp.add_argument("--split", default="test", choices=["val", "test"])
    pp.add_argument("--weights", default="ema", choices=["ema", "model"])
    pp.add_argument("--out", type=Path, required=True)
    pp.add_argument("--limit-videos", type=int)
    pp.add_argument("--set", nargs="*", default=[], help="config overrides a.b=c")
    sp = sub.add_parser("score")
    sp.add_argument("--pred", type=Path, required=True)
    sp.add_argument("--index-dir", type=Path, required=True)
    sp.add_argument("--split", default="test", choices=["val", "test"])
    sp.add_argument("--flocks", type=Path, help="JSON from scripts/flag_flocks.py")
    sp.add_argument("--presence-threshold", type=float,
                    help="use presence_score >= this; default: mask area > --eps")
    sp.add_argument("--eps", type=int, default=0, help="pixels a mask needs to count as present")
    sp.add_argument("--workers", type=int, default=8)
    sp.add_argument("--out", type=Path, required=True)
    sp.add_argument("--mlflow-run-id", help="also log the overall metrics to this MLflow run")
    args = p.parse_args()

    if args.cmd == "predict":
        from utils.config import load_config
        predict(load_config(args.config, args.set), args.ckpt, args.split, args.out,
                args.weights, args.limit_videos)
        return

    split_file = "train" if args.split == "val" else "test"
    frames = pd.read_parquet(args.index_dir / f"frames_{split_file}.parquet")
    frames = frames[frames["split"] == args.split]
    inst = pd.read_parquet(args.index_dir / f"instances_{split_file}.parquet")
    inst = inst[inst["split"] == args.split]
    pred = pd.read_parquet(args.pred)
    for c in PRED_COLUMNS:
        if c not in pred:
            pred[c] = np.nan
    flocks = set(json.loads(args.flocks.read_text())["flagged_videos"]) if args.flocks else None
    df, inst_df = frame_metrics(pred, frames, inst, args.workers)
    res = strata(df, inst_df, flocks, args.presence_threshold, args.eps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, default=float))
    df.drop(columns=["union_rle", "pred_rle", "inst_bboxes", "inst_areas"], errors="ignore") \
      .to_parquet(args.out.with_suffix(".frames.parquet"), index=False)
    print(json.dumps(res["overall"], indent=1))
    if args.mlflow_run_id:
        import mlflow
        from utils.tracking import tracking_uri
        mlflow.set_tracking_uri(tracking_uri())
        with mlflow.start_run(run_id=args.mlflow_run_id):
            mlflow.log_metrics({f"{args.split}/{k}": v for k, v in res["overall"].items()
                                if isinstance(v, (int, float)) and not np.isnan(v)})
            mlflow.log_artifact(str(args.out))


if __name__ == "__main__":
    main()
