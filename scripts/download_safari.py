"""Download SA-FARI annotations (Hugging Face) and the 6 fps frame folders (GCS).

Annotations come from the gated Hugging Face dataset facebook/SA-FARI, so run
`huggingface-cli login` with an account that has accepted the terms first.

Frames come from the public bucket gs://cxl-public-camera-trap named on the dataset
card. Only `JPEGImages_6fps` is fetched; the full-rate `videos/` folders are never
touched. The script lists the tarballs, prints the expected size and asks for
confirmation before downloading any frames.

Examples
--------
    # annotations only (about 1.9 GB)
    python scripts/download_safari.py --root $SAFARI_ROOT --annotations

    # test frames, list sizes and ask before downloading, then extract
    python scripts/download_safari.py --root $SAFARI_ROOT --frames test --extract

    # a few loose test videos for a smoke test, no tarballs
    python scripts/download_safari.py --root $SAFARI_ROOT --sample-videos 3
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import shutil
import sys
import tarfile
from pathlib import Path

import requests

HF_REPO = "facebook/SA-FARI"
ANNOTATION_FILES = [
    "annotation/sa_fari_train.json",
    "annotation/sa_fari_test.json",
    "annotation/sa_fari_train_ext.json",
    "annotation/sa_fari_test_ext.json",
]
BUCKET = "cxl-public-camera-trap"
LIST_URL = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
OBJECT_URL = f"https://storage.googleapis.com/{BUCKET}/{{name}}"
TAR_PREFIX = "sa_fari/sa_fari_{split}_tars/JPEGImages_6fps/"
LOOSE_PREFIX = "sa_fari/sa_fari_{split}/JPEGImages_6fps/"


def list_objects(prefix: str) -> list[dict]:
    items, token = [], None
    while True:
        params = {"prefix": prefix, "fields": "items(name,size,md5Hash),nextPageToken",
                  "maxResults": 1000}
        if token:
            params["pageToken"] = token
        r = requests.get(LIST_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        items += [i for i in data.get("items", []) if not i["name"].endswith("/")]
        token = data.get("nextPageToken")
        if not token:
            return items


def download_object(obj: dict, dest: Path) -> None:
    """Stream one object to disk with resume support and an md5 check."""
    size = int(obj["size"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and dest.stat().st_size == size:
        print(f"  skip {dest.name} (already complete)")
        return
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    with requests.get(OBJECT_URL.format(name=obj["name"]), headers=headers, stream=True,
                      timeout=120) as r:
        r.raise_for_status()
        if start and r.status_code != 206:
            start = 0  # server ignored the range, restart
        with part.open("ab" if start else "wb") as f:
            done = start
            for chunk in r.iter_content(chunk_size=8 << 20):
                f.write(chunk)
                done += len(chunk)
                print(f"\r  {dest.name}: {done / 1e9:6.2f} / {size / 1e9:.2f} GB", end="",
                      flush=True)
    print()
    if "md5Hash" in obj:
        md5 = hashlib.md5()
        with part.open("rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                md5.update(chunk)
        if base64.b64encode(md5.digest()).decode() != obj["md5Hash"]:
            raise RuntimeError(f"md5 mismatch for {dest}; delete {part} and retry")
    part.rename(dest)


def confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        print(f"{prompt} [--yes given]")
        return True
    if not sys.stdin.isatty():
        print("No terminal to confirm on. Re-run interactively or pass --yes.")
        return False
    return input(f"{prompt} [y/N] ").strip().lower() in {"y", "yes"}


def download_annotations(root: Path) -> None:
    from huggingface_hub import hf_hub_download

    for f in ANNOTATION_FILES:
        path = hf_hub_download(HF_REPO, f, repo_type="dataset", local_dir=root)
        print(f"  {path}")


def download_frames(root: Path, splits: list[str], extract: bool, assume_yes: bool) -> None:
    plan = []
    for split in splits:
        objs = list_objects(TAR_PREFIX.format(split=split))
        if not objs:
            raise RuntimeError(f"No tarballs under {TAR_PREFIX.format(split=split)}")
        plan += [(split, o) for o in objs]
    total = sum(int(o["size"]) for _, o in plan)
    print("Frame tarballs (JPEGImages_6fps only):")
    for split, o in plan:
        print(f"  {split:5s} {o['name'].rsplit('/', 1)[-1]:28s} {int(o['size']) / 1e9:7.2f} GB")
    print(f"Expected download: {total / 1e9:.2f} GB in {len(plan)} files"
          + (" (extraction needs about the same again)" if extract else ""))
    free = shutil.disk_usage(root).free
    print(f"Free space under {root}: {free / 1e9:.1f} GB")
    if free < total * (2 if extract else 1):
        print("WARNING: not enough free space for this download.")
    if not confirm("Download these frames?", assume_yes):
        print("Aborted, nothing downloaded.")
        return
    for split, o in plan:
        tar_path = root / "tars" / split / o["name"].rsplit("/", 1)[-1]
        download_object(o, tar_path)
        if extract:
            out = root / f"sa_fari_{split}"
            out.mkdir(parents=True, exist_ok=True)
            print(f"  extracting into {out}")
            with tarfile.open(tar_path) as tf:
                tf.extractall(out, filter="data")


def download_sample(root: Path, split: str, n_videos: int) -> None:
    """Fetch the loose frames of the first n videos of a split, for smoke tests."""
    prefix = LOOSE_PREFIX.format(split=split)
    videos: dict[str, list[dict]] = {}
    token = None
    while len(videos) <= n_videos:
        params = {"prefix": prefix, "fields": "items(name,size,md5Hash),nextPageToken",
                  "maxResults": 1000}
        if token:
            params["pageToken"] = token
        data = requests.get(LIST_URL, params=params, timeout=60).json()
        for o in data.get("items", []):
            rel = o["name"][len(prefix):]
            if "/" in rel:
                videos.setdefault(rel.split("/")[0], []).append(o)
        token = data.get("nextPageToken")
        if not token:
            break
    names = sorted(videos)[:n_videos]
    objs = [o for v in names for o in videos[v]]
    print(f"Sample: {len(names)} videos, {len(objs)} frames, "
          f"{sum(int(o['size']) for o in objs) / 1e6:.1f} MB")
    for o in objs:
        rel = o["name"][len(prefix):]
        dest = root / f"sa_fari_{split}" / "JPEGImages_6fps" / rel
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(OBJECT_URL.format(name=o["name"]), timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="Dataset root on shared storage")
    p.add_argument("--annotations", action="store_true", help="Download the four annotation files")
    p.add_argument("--frames", nargs="+", choices=["train", "test"], default=[],
                   help="Download JPEGImages_6fps tarballs for these splits")
    p.add_argument("--extract", action="store_true", help="Extract tarballs after download")
    p.add_argument("--sample-videos", type=int, default=0,
                   help="Download loose frames for the first N test videos only")
    p.add_argument("--yes", action="store_true",
                   help="Skip the confirmation prompt (non-interactive batch jobs only)")
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.annotations:
        print("Annotations from Hugging Face:")
        download_annotations(args.root)
    if args.frames:
        download_frames(args.root, args.frames, args.extract, args.yes)
    if args.sample_videos:
        download_sample(args.root, "test", args.sample_videos)
    if not (args.annotations or args.frames or args.sample_videos):
        p.print_help()


if __name__ == "__main__":
    main()
