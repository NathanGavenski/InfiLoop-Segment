"""Phase 1 resolution study.

For candidate long sides (768, 1024, 1344, 1536 px) and patch sizes (14, 16), count the
instances that would cover less than one patch and less than four patches after resizing
the frame by its long side. The proposed rule picks, per patch size, the smallest long side
at which at least 99% of instances cover one patch or more.

An "instance" is one masklet in one frame (one row of the instance index), and its
coverage is its mask area in patches: area * (long_side / max(H, W))^2 / patch^2.

Reads either annotation JSON files (RLE strings are dropped while parsing, so the 0.9 GB
train file fits in modest memory) or instance-index Parquet files.

    python scripts/resolution_study.py $SAFARI_ROOT/annotation/sa_fari_train_ext.json \\
        $SAFARI_ROOT/annotation/sa_fari_test_ext.json --out outputs/resolution_study.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

LONG_SIDES = (768, 1024, 1344, 1536)
PATCHES = (14, 16)
TARGET = 0.99


def _drop_rle(obj: dict):
    return None if "counts" in obj and "size" in obj else obj


def instances_from_json(path: Path) -> pd.DataFrame:
    with path.open() as f:
        data = json.load(f, object_hook=_drop_rle)
    videos = {v["id"]: v for v in data["videos"]}
    rows = []
    for a in data["annotations"]:
        v = videos[a["video_id"]]
        for area in a["areas"]:
            if area is not None:
                rows.append((area, v["height"], v["width"]))
    df = pd.DataFrame(rows, columns=["area", "height", "width"])
    df["source"] = path.stem
    return df


def instances_from_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["area", "height", "width"])
    df["source"] = path.stem
    return df


def study(df: pd.DataFrame, long_sides=LONG_SIDES, patches=PATCHES) -> pd.DataFrame:
    area = df["area"].to_numpy(float)
    long = np.maximum(df["height"].to_numpy(float), df["width"].to_numpy(float))
    rows = []
    for patch in patches:
        for L in long_sides:
            cov = area * (L / long) ** 2 / patch ** 2
            rows.append({
                "patch": patch, "long_side": L, "instances": len(cov),
                "lt1_patch": int((cov < 1).sum()), "lt4_patches": int((cov < 4).sum()),
                "pct_ge1_patch": 100 * float((cov >= 1).mean()),
                "pct_ge4_patches": 100 * float((cov >= 4).mean()),
                "tokens_16x9": math.ceil(L * 9 / 16 / patch) * math.ceil(L / patch),
            })
    return pd.DataFrame(rows)


def choose(table: pd.DataFrame, target: float = TARGET) -> dict[int, int | None]:
    out = {}
    for patch, g in table.groupby("patch"):
        ok = g[g.pct_ge1_patch >= 100 * target].sort_values("long_side")
        out[int(patch)] = int(ok.long_side.iloc[0]) if len(ok) else None
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", type=Path, help="annotation JSON or instance Parquet")
    p.add_argument("--out", type=Path, help="write the table as CSV")
    p.add_argument("--target", type=float, default=TARGET)
    args = p.parse_args()

    dfs = [instances_from_parquet(x) if x.suffix == ".parquet" else instances_from_json(x)
           for x in args.inputs]
    frames = {"all": pd.concat(dfs, ignore_index=True)}
    if len(dfs) > 1:
        frames.update({d.source.iloc[0]: d for d in dfs})
    tables = []
    for name, d in frames.items():
        t = study(d)
        t.insert(0, "subset", name)
        tables.append(t)
        print(f"\n== {name}: {len(d)} instances, frame long side "
              f"{int(np.maximum(d.height, d.width).min())}-{int(np.maximum(d.height, d.width).max())} px")
        print(t.drop(columns="subset").to_string(index=False, float_format="%.2f"))
        print(f"Smallest long side with >= {100 * args.target:.0f}% of instances covering "
              f">= 1 patch: {choose(t, args.target)}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(tables).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
