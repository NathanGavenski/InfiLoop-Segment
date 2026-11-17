"""Train the Phase 2 baselines B0 (linear probe) and B1 (single-pass decoder).

    python train.py --config configs/b1_dinov3.yaml [--set train.epochs=5 ...]

Each epoch draws a few frames per video (stratified so empty frames are included), groups
them into aspect-ratio buckets and trains with BCE + Dice at mask resolution, ignoring
padded pixels. An EMA of the trainable weights is kept, and checkpoints hold both the raw
and EMA weights (encoder weights are frozen and not saved). The run logs its config, git
commit, seed and data split hash to MLflow.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.augment import AugmentConfig
from data.buckets import make_bucket_batches
from data.dataset import FrameDataset, collate
from data.sa_fari import sample_frames
from models.segmenter import build_model, trainable_parameters
from utils.config import load_config
from utils.tracking import seed_everything, split_hash, start_run


# --------------------------------------------------------------------------- loss

def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                  bce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1.0):
    """BCE + soft Dice over valid (non-padded) pixels. logits, target, valid: BxHxW.

    Dice is per image with +1 smoothing, so an empty target with an empty prediction costs
    nothing and an empty target with false positives is still penalised.
    """
    v = valid.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * v).sum() / v.sum().clamp_min(1)
    p = torch.sigmoid(logits) * v
    t = target * v
    inter = (p * t).flatten(1).sum(1)
    denom = p.flatten(1).sum(1) + t.flatten(1).sum(1)
    dice = 1 - (2 * inter + smooth) / (denom + smooth)
    return bce_weight * bce + dice_weight * dice.mean(), {"bce": bce.item(),
                                                         "dice": dice.mean().item()}


# --------------------------------------------------------------------------- EMA

class EMA:
    """EMA of the trainable (non-encoder) parameters and buffers, as in TRM."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in self._state(model).items()}

    @staticmethod
    def _state(model):
        return {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}

    @torch.no_grad()
    def update(self, model):
        for k, v in self._state(model).items():
            if v.dtype.is_floating_point:
                self.shadow[k].lerp_(v.detach(), 1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}


# --------------------------------------------------------------------------- data

def load_frames(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = cfg["data"]
    frames = pd.read_parquet(Path(d["index_dir"]) / "frames_train.parquet")
    train = frames[frames["split"] == "train"].reset_index(drop=True)
    val = frames[frames["split"] == "val"].reset_index(drop=True)
    # Fixed validation subset: the same frames every epoch and every run with this seed.
    k = d.get("val_frames_per_video", 4)
    val_idx = sample_frames(val, k, d.get("empty_share", 0.25), np.random.default_rng(cfg["seed"]))
    return train, val.loc[np.sort(val_idx)].reset_index(drop=True)


def make_loader(ds: FrameDataset, indices, batch_size: int, rng, num_workers: int, shuffle: bool):
    batches = make_bucket_batches(indices, ds.keys, batch_size, rng if shuffle else None)
    return DataLoader(ds, batch_sampler=batches, collate_fn=collate, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=False)


# --------------------------------------------------------------------------- validation

@torch.no_grad()
def validate(model, loader, device, amp_dtype) -> dict:
    """IoU at training resolution on the fixed validation subset. Final numbers come from
    eval.py at original resolution; this is for model selection only."""
    model.eval()
    inter = union = 0.0
    ious, losses = [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device)
        valid = batch["valid"].to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(images)["logits"][:, 0].float()
        loss, _ = bce_dice_loss(logits, mask, valid)
        losses.append(loss.item())
        pred = (logits > 0) & valid
        tgt = (mask > 0.5) & valid
        i = (pred & tgt).flatten(1).sum(1).float()
        u = (pred | tgt).flatten(1).sum(1).float()
        ious += torch.where(u > 0, i / u.clamp_min(1), torch.ones_like(u)).tolist()
        inter += i.sum().item()
        union += u.sum().item()
    model.train()
    return {"val/loss": float(np.mean(losses)), "val/mean_iou": float(np.mean(ious)),
            "val/global_iou": inter / max(union, 1)}


# --------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", nargs="*", default=[], help="config overrides a.b=c")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    seed = cfg["seed"]
    seed_everything(seed)
    d, t = cfg["data"], cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16

    train_df, val_df = load_frames(cfg)
    model = build_model(cfg).to(device)
    patch = model.encoder.patch_size
    aug = AugmentConfig(**cfg.get("augment", {})) if cfg.get("augment") is not None else None
    train_ds = FrameDataset(train_df, d["image_root_train"], d["long_side"], patch,
                            d.get("square", False), aug, seed)
    val_ds = FrameDataset(val_df, d["image_root_train"], d["long_side"], patch, d.get("square", False))
    val_loader = make_loader(val_ds, range(len(val_ds)), t["batch_size"], None,
                             d.get("num_workers", 4), shuffle=False)

    params = trainable_parameters(model)
    opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t.get("weight_decay", 0.05),
                            betas=tuple(t.get("betas", (0.9, 0.95))))
    est_steps = t["epochs"] * math.ceil(train_df.video_id.nunique() * d["frames_per_video"]
                                        / t["batch_size"])
    warmup = t.get("warmup_steps", 200)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, s / max(est_steps, 1)))))
    ema = EMA(model, t.get("ema_decay", 0.999))

    hashes = split_hash({"train": train_df.video_id.unique(), "val": val_df.video_id.unique(),
                         "val_frames": val_df.video_id.astype(str) + ":" + val_df.frame_idx.astype(str)})
    out_dir = Path(cfg.get("output_dir", "outputs")) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{cfg['name']}: {sum(p.numel() for p in params) / 1e6:.2f}M trainable params, "
          f"{train_df.video_id.nunique()} train videos, {len(val_df)} val frames, device {device}")

    import mlflow
    with start_run(cfg, seed, hashes) as run:
        mlflow.log_param("trainable_params", sum(p.numel() for p in params))
        best, step = -1.0, 0
        rng = np.random.default_rng(seed)
        ema_model = copy.deepcopy(model)
        for epoch in range(t["epochs"]):
            train_ds.set_epoch(epoch)
            idx = sample_frames(train_df, d["frames_per_video"], d.get("empty_share", 0.25), rng)
            loader = make_loader(train_ds, idx, t["batch_size"], rng, d.get("num_workers", 4), True)
            model.train()
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                valid = batch["valid"].to(device, non_blocking=True)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                    out = model(images)
                loss, parts = bce_dice_loss(out["logits"][:, 0].float(), mask, valid,
                                            t.get("bce_weight", 1.0), t.get("dice_weight", 1.0))
                if t.get("coarse_weight", 0.0):
                    coarse = F.interpolate(out["coarse"].float(), size=mask.shape[-2:],
                                           mode="bilinear", align_corners=False)[:, 0]
                    closs, _ = bce_dice_loss(coarse, mask, valid)
                    loss = loss + t["coarse_weight"] * closs
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if t.get("grad_clip"):
                    torch.nn.utils.clip_grad_norm_(params, t["grad_clip"])
                opt.step()
                sched.step()
                ema.update(model)
                step += 1
                if step % t.get("log_every", 50) == 0:
                    mlflow.log_metrics({"train/loss": loss.item(), **{f"train/{k}": v for k, v in parts.items()},
                                        "lr": sched.get_last_lr()[0]}, step=step)
            if (epoch + 1) % t.get("val_every", 1) == 0 or epoch + 1 == t["epochs"]:
                ema_model.load_state_dict(ema.state_dict(), strict=False)
                metrics = validate(ema_model, val_loader, device, amp_dtype)
                mlflow.log_metrics({**metrics, "epoch": epoch}, step=step)
                print(f"epoch {epoch} step {step} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                ckpt = {"model": {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")},
                        "ema": ema.state_dict(), "cfg": cfg, "epoch": epoch, "step": step,
                        "metrics": metrics, "data_split_hash": hashes, "mlflow_run_id": run.info.run_id}
                torch.save(ckpt, out_dir / "last.ckpt")
                if metrics["val/mean_iou"] > best:
                    best = metrics["val/mean_iou"]
                    torch.save(ckpt, out_dir / "best.ckpt")
                    mlflow.log_metric("val/best_mean_iou", best, step=step)
        print(f"done; best val mean IoU {best:.4f}; checkpoints in {out_dir}; "
              f"MLflow run {run.info.run_id}")


if __name__ == "__main__":
    main()
