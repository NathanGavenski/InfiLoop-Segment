# InfiLoop-Segment (camtrap-loop)

Per-frame animal foreground segmentation for camera trap video. A frozen pretrained encoder
supplies patch features, an InfiLoop recursive block refines a foreground mask over the patch
grid, and the TRM halting head doubles as the animal presence gate. See [PLAN.md](PLAN.md) for
the full strategy.

**Status:** Phases 0 to 2 (environment, SA-FARI data pipeline, baselines B0, B1 and B2). The
recursive block and the halting gate (Phases 3 and 4) are not implemented yet.

## Licences and terms of use

| Asset | Licence | Consequence |
| --- | --- | --- |
| SA-FARI annotations and frames (Meta and CXL) | CC-BY-NC 4.0 | Non-commercial use only, attribution required |
| SAM 3 / SAM 3.1 weights | SAM License (gated on Hugging Face) | Used for baseline B2 and Stage B pseudo-labels only, under its terms |
| DINOv3 weights (default encoder) | DINOv3 License (gated on Hugging Face) | Check terms before committing to it as the encoder |
| BioCLIP 2 weights | Check the model card before use | Candidate encoder |
| InfiLoop and TRM code | See each upstream repository | Reference only, kept in `third_party/` |

**Every Stage A artefact (checkpoints, predictions, metrics, figures) is derived from SA-FARI and
is therefore research-only and non-commercial.** Do not ship Stage A checkpoints in a product.

Attribution: SA-FARI dataset, Meta and Conservation X Labs,
<https://huggingface.co/datasets/facebook/SA-FARI>.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/): `pyproject.toml` declares
them and `uv.lock` pins every version. On the HPC everything runs inside an
Apptainer/Singularity image built from that lock; the code itself is bound in at run time,
so only dependency changes need a rebuild.

```bash
# Local development
uv sync --extra sam3          # .venv with the locked versions (drop --extra sam3 if not needed)
uv run pytest

# Container: build where you have root or fakeroot (about 7 GB), then copy it to the HPC
bash container/build.sh camtrap.sif          # or: sudo -E bash container/build.sh camtrap.sif
bash scripts/hpc/push_image.sh              # rsync to create:$WORK_DIR/containers/, resumable, checks sha256

# On the KCL HPC (paths are set in scripts/hpc/env.sh)
source scripts/hpc/env.sh                    # defines `run` = apptainer exec --nv ... camtrap.sif
bash scripts/fetch_upstream.sh               # InfiLoop and TRM references into third_party/

# Data: accept the terms on Hugging Face and run `huggingface-cli login` first
# run_data = uv env without torch/CUDA (scripts/hpc/requirements-data.txt); needs uv, not the image
run_data python scripts/download_safari.py --root $SAFARI_ROOT --annotations
run_data python scripts/download_safari.py --root $SAFARI_ROOT --frames test train --extract   # asks first
```

To add or change a dependency: `uv add <pkg>==<version>` (or edit `pyproject.toml` and run
`uv lock`), commit both files, and rebuild the image. Jobs print a warning when the image was
built from a different `uv.lock`.

The 6 fps frame tarballs are about 3.7 GB (test) and 60.8 GB (train). The script prints the
exact size and asks for confirmation before downloading any frames. Point
`data.image_root_*` in `configs/base.yaml` at the folder holding the `sa_fari_XXXXXX/`
video folders after extraction.

## Running the Phase 1 and 2 pipeline

```bash
mkdir -p logs                                         # SLURM output directory
sbatch scripts/hpc/prepare_data.sh                    # frame tables, instance index, resolution study
sbatch scripts/hpc/train.sh configs/b0_dinov3.yaml    # B0 (also b0_bioclip2, b0_sam3)
sbatch scripts/hpc/train.sh configs/b1_dinov3.yaml    # B1
sbatch scripts/hpc/eval.sh  configs/b1_dinov3.yaml val
sbatch scripts/hpc/b2_sam3.sh                         # B2, sharded; score with eval.py afterwards
```

Every method writes the same per-frame prediction file, and `eval.py score` computes all
metrics and strata from it. The validation split (32 of 651 training locations, 5.7% of
frames) is fixed in `splits/sa_fari_val_locations.json`. The Phase 1 resolution study on
the full annotations is in `reports/resolution_study.csv`.

Local smoke test without a GPU or real weights: `configs/smoke_b1.yaml` uses a tiny randomly
initialised encoder (see the comments in that file).

## Next steps

Phases 0 to 2 are written and unit-tested, but nothing has run on the HPC or with real
weights yet. In order:

- [x] **Build the image.** `container/build.sh` needs root or fakeroot. Where neither is
  allowed, run `sudo -E bash container/build.sh camtrap.sif` on a Linux machine (about 7 GB)
  and copy it to `$WORK_DIR/containers/` on the HPC. Try a rootless build on the HPC first.
- [x] **Request access on Hugging Face** to the gated weights:
  `facebook/dinov3-vitl16-pretrain-lvd1689m`, `facebook/sam3`, and `facebook/sam3.1` if the
  3.1 variant of B2 is wanted.
- [x] **Set up the HPC.** Fill in the `CHANGE_ME` paths in `scripts/hpc/env.sh`, run
  `huggingface-cli login`, and `mkdir -p logs`.
- [ ] **Get the data.** Download the annotations and the 6 fps frames (64 GB; the script asks
  first) with `run_data`, no SLURM needed. Then index the annotations on a compute node:
  `sbatch scripts/hpc/prepare_data.sh`, or `bash` it inside an `interactive_cpu` session.
  Indexing needs only the annotations, so it can run while the frames download.
- [ ] **Smoke-test on a GPU** before any long job, for example
  `sbatch scripts/hpc/train.sh configs/b0_dinov3.yaml train.epochs=1`. This checks the image,
  the `--nv` GPU mount and the real weights.
- [ ] **Choose the encoder.** Train B0 with DINOv3, BioCLIP 2 and SAM 3, and compare them on
  validation. Train B1 with the winner (`configs/b1_dinov3.yaml` assumes DINOv3).
- [ ] **Run B2.** `sbatch scripts/hpc/b2_sam3.sh`, then `scripts/flag_flocks.py` on its
  predictions, then score every method with the flock flags.
- [ ] **Review the baseline table** (B0, B1, B2 on the same test frames) before starting
  Phase 3, the recursive block.

Open decisions:

- **SAM 3 or SAM 3.1 for B2.** SAM 3.1 is released as a video tracking checkpoint, while B2
  runs the image model per frame, so B2 defaults to SAM 3 weights. `sam3.version: sam3.1`
  loads 3.1's detector into the image model instead; that path is untested, so check the
  loading log for missing weights before trusting it.
- **GPU constraint.** Pin the A100 type in the SLURM scripts (`#SBATCH --constraint=...`) so
  latency per frame is comparable across methods.

## MLflow

Runs log to the file store given by `MLFLOW_TRACKING_URI` (set in `scripts/hpc/env.sh` to
`$WORK_DIR/mlruns`; it defaults to `./mlruns` locally). Each run records its config, git commit,
seed and data split hash. Browse with `mlflow ui --backend-store-uri $MLFLOW_TRACKING_URI`.

## Layout

```
configs/        one YAML per experiment
data/           SA-FARI parsing, union masks, instance index, bucketing, augmentation
models/         frozen encoder wrapper, decoder, heads, baseline assembly
train.py        trains B0 and B1
eval.py         prediction export, metrics and strata
scripts/        downloads, resolution study, B2 (SAM 3), flock flags, HPC job scripts
tests/          unit tests (run with `pytest`)
```
