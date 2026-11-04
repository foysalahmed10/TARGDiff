<div align="center">

# TARGDiff: Time-Adaptive Routing-Gated Diffusion for 3D Point Cloud Completion

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Task](https://img.shields.io/badge/Task-3D%20Point%20Cloud%20Completion-6f42c1)](#overview)

**Research code for conditional 3D point-cloud completion.**

</div>

## Overview

TARGDiff is a diffusion-based framework for completing partial 3D point clouds. This repository provides the model implementation and the tools for dataset preparation, training, evaluation, and inference.

The repository is intended to accompany the TARGDiff research paper. Author and publication details will be added when they are ready for public release.

## Visual examples

<p align="center">
  <img src="assets/overview.png" alt="High-level illustration of partial point-cloud completion with TARGDiff" width="96%">
</p>
<p align="center"><em>High-level illustration of the point-cloud completion task.</em></p>

<p align="center">
  <img src="assets/dataset_protocol.png" alt="Dataset examples across object categories and missing-geometry levels" width="96%">
</p>
<p align="center"><em>Examples of the dataset at different missing-geometry levels.</em></p>

<p align="center">
  <img src="assets/qualitative_75_missing.png" alt="Qualitative point-cloud completion comparison at 75 percent missing geometry" width="96%">
</p>
<p align="center"><em>Qualitative comparison in the 75% missing-geometry setting.</em></p>

## Languages and technologies

| Language or format | Use |
|---|---|
| **Python** | Model, data processing, training, evaluation, inference, and tests |
| **Markdown** | Documentation |
| **JSON** | Experiment configuration |
| **YAML** | Reference environment configuration |
| **TOML** | Python package and build configuration |

**Main libraries:** PyTorch, NumPy, and pytest.

## Installation

Python 3.10 is recommended. Install a PyTorch version compatible with your operating system, GPU, and CUDA driver by following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

Then install TARGDiff:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

An optional Conda environment reference is provided in [`environment-reference.yml`](environment-reference.yml). Review the PyTorch/CUDA selection for your system before creating an environment from it.

## Dataset preparation

Datasets are not included in this repository. Supported point-cloud files are `.txt`, `.xyz`, `.pts`, and `.npy`. Place files in one directory per object class. See [`data/README.md`](data/README.md) for the expected layout and file requirements.

Create a source-level split manifest:

```bash
python scripts/make_split_manifest.py \
  --input-roots data/raw/ClassA data/raw/ClassB \
  --output data/splits/source_split.csv
```

Prepare the 25%, 50%, and 75% missing-geometry datasets:

```bash
python scripts/prepare_dataset.py \
  --split-manifest data/splits/source_split.csv \
  --output-root data/prepared \
  --remove-ratios 0.25 0.50 0.75 \
  --full-points 2048 \
  --partial-points 2048 \
  --structure-points 256
```

The prepared data are written to separate `remove_25`, `remove_50`, and `remove_75` directories, each containing `train`, `val`, and `test` splits.

## Training

Example training command for the 50% missing-geometry setting:

```bash
python scripts/train.py \
  --config configs/paper_controlled_2048.json \
  --train-root data/prepared/remove_50/train \
  --val-root data/prepared/remove_50/val \
  --output runs/remove_50 \
  --device cuda:0
```

The example uses the supplied configuration. The best checkpoint is saved at `runs/remove_50/checkpoints/best.pt`.

## Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint runs/remove_50/checkpoints/best.pt \
  --data-root data/prepared/remove_50/test \
  --output evaluations/remove_50 \
  --device cuda:0
```

## Inference

Use a trained checkpoint and a partial point cloud in the same coordinate and preprocessing convention used during training:

```bash
python scripts/infer.py \
  --checkpoint runs/remove_50/checkpoints/best.pt \
  --partial path/to/partial_cloud.xyz \
  --output outputs/completed_cloud.txt \
  --device cuda:0
```

Add `--normalize-partial` only when normalization is required to match the training data preprocessing.

## Repository structure

```text
TARGDiff/
├── assets/       # Selected overview and qualitative figures
├── configs/      # Experiment configurations
├── data/         # Dataset instructions (data are not included)
├── scripts/      # Dataset, training, evaluation, and inference tools
├── src/targdiff/ # TARGDiff implementation
├── tests/        # Automated tests
├── README.md
└── pyproject.toml
```

## Tests

```bash
pytest
python -m compileall -q src scripts
```

## License

A license has not yet been selected. Please do not assume that the code is available for reuse, modification, or redistribution. See [`LICENSE_NOT_SELECTED.md`](LICENSE_NOT_SELECTED.md).

## Citation

Citation details will be added when the paper is publicly available.
