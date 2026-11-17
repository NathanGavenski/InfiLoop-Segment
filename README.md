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

```bash
# On the KCL HPC: edit the CHANGE_ME paths in scripts/hpc/env.sh, then
bash scripts/hpc/build_env.sh            # add --sam3 for baseline B2
source scripts/hpc/env.sh

# Upstream references (InfiLoop and TRM) into third_party/
bash scripts/fetch_upstream.sh

# Data: accept the terms on Hugging Face and run `huggingface-cli login` first
python scripts/download_safari.py --root $SAFARI_ROOT --annotations
python scripts/download_safari.py --root $SAFARI_ROOT --frames test train --extract   # asks first
```

The 6 fps frame tarballs are about 3.7 GB (test) and 60.8 GB (train). The script prints the
exact size and asks for confirmation before downloading any frames.

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
