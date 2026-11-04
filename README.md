<div align="center">

# TARGDiff

### Time-Adaptive Routing-Gated Diffusion for 3D Point Cloud Completion

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Task](https://img.shields.io/badge/Task-3D%20Point%20Cloud%20Completion-6A5ACD.svg)](#)

**Official implementation of TARGDiff.**

</div>

<p align="center">
  <img src="assets/fig1_overview.png" alt="Overview of TARGDiff and the TARGFuse module" width="96%">
</p>

## Overview

TARGDiff is a conditional point-space diffusion framework for 3D point cloud completion. It represents the observed geometry with two complementary views: the dense partial point cloud and a 256-point farthest-point-sampled (FPS) subset of that same observation. Their global features jointly condition the denoiser, while their local features remain separate and are aligned to the current denoising query.

The proposed **TARGFuse** module predicts token-wise routing weights from the query, the two aligned contexts, and the diffusion timestep. It fuses the contexts and then uses their disagreement for residual refinement. The structural condition is derived only from the partial input:

\[
S = \operatorname{FPS}_{256}(P).
\]

## Method

<p align="center">
  <img src="assets/fig2_architecture.png" alt="TARGDiff architecture and training flow" width="96%">
</p>

For each denoising query, TARGFuse computes two query-aligned contexts, one from the dense observation and one from the FPS view. A two-way softmax assigns their relative contributions:

\[
[w_p,w_s]=\operatorname{softmax}(\operatorname{MLP}_g([Q,C_p,C_s,t_{tok}])).
\]

The routed context is refined using the symmetric disagreement descriptor \(D=|C_p-C_s|\):

\[
C_f=w_p\odot C_p+w_s\odot C_s,\qquad
Q_f=Q+\phi([C_f,D]).
\]

The routing weights indicate relative branch preference; they are **not calibrated uncertainty estimates**. The complete target is used for training supervision only and is not an inference-time condition.

## Results

### Controlled missingness

The manuscript evaluates separately trained models at 25%, 50%, and 75% missing geometry using a PCN-derived directional-cropping protocol.

| Missing geometry | CD-L2 (\(\times 10^4\)) ↓ | F-score@0.02 ↑ |
|---:|---:|---:|
| 25% | **8.71** | **0.896** |
| 50% | **12.43** | **0.760** |
| 75% | **17.34** | **0.573** |

The manuscript reports CD-L2 reductions of 3.5%, 5.3%, and 12.0%, respectively, over the strongest listed CD baseline at each missingness level.

### Additional benchmarks

- **Completion3D:** average CD of **7.14** (reported using the benchmark's \(10^{-4}\) scale).
- **ShapeNet-55:** CD-L2 of **1.08** on the Hard setting (reported using the benchmark's \(10^{-3}\) scale).

<p align="center">
  <img src="assets/fig5_qualitative_50.png" alt="Qualitative point cloud completion results at 50 percent missing geometry" width="96%">
</p>

<p align="center"><em>Qualitative comparison at 50% missing geometry. See <a href="assets/fig4_qualitative_25.png">25%</a> and <a href="assets/fig6_qualitative_75.png">75%</a> results.</em></p>

> **Result provenance:** Numerical results above are transcribed from the accompanying manuscript. The exact paper checkpoints and audited source split manifest are not included in this code release; these values are paper-reported and are not claimed to have been independently reproduced from this repository snapshot.

## Installation

```bash
git clone https://github.com/foysalahmed10/TARGDiff.git
cd TARGDiff
python -m pip install -e .
```

The package requires Python 3.10 or later, NumPy, and PyTorch. For development and tests:

```bash
python -m pip install -r requirements-dev.txt
```

## Data preparation

Benchmark datasets are not redistributed. Prepare the source-level train/test split **before** generating partial observations to prevent object-level leakage. For the main method, derive the structure input from the partial cloud (`S = FPS_256(P)`).

```bash
python scripts/prepare_dataset.py \
  --input-roots data/raw/Airplane data/raw/Cabinet data/raw/Car \
                data/raw/Chair data/raw/Lamp data/raw/Sofa \
                data/raw/Table data/raw/Watercraft \
  --output-root data/prepared \
  --remove-ratios 0.25 0.50 0.75 \
  --full-points 2048 \
  --partial-points 2048 \
  --structure-points 256 \
  --sampler fps \
  --structure-source partial \
  --seed 42
```

For exact paper reproduction, use the original audited source split manifest. A newly generated split is a new experiment, even when the same seed is used. See [`data/README.md`](data/README.md) for the expected data layout and [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md) for details.

## Training

The manuscript-aligned controlled-experiment configuration is:

`configs/paper_manuscript_controlled_2048.json`

It specifies 2,048 partial points, 256 structure points, 400 diffusion steps, epsilon prediction, a linear beta schedule, fixed-beta reverse variance, and 400 training epochs.

Example (50% missing geometry):

```bash
python scripts/train.py \
  --config configs/paper_manuscript_controlled_2048.json \
  --train-root data/prepared/remove_50/train \
  --output runs/paper/remove_50 \
  --device cuda:0
```

Train separate models for the 25%, 50%, and 75% settings by changing the data and output paths. See [`docs/PAPER_REPRODUCTION.md`](docs/PAPER_REPRODUCTION.md) for the full protocol and configuration details.

## Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint runs/paper/remove_50/checkpoints/epoch_0400.pt \
  --data-root data/prepared/remove_50/test \
  --output evaluations/remove_50 \
  --device cuda:0 \
  --weights raw \
  --seed 42 \
  --f-threshold 0.02 \
  --cd-scale 10000
```

The evaluator uses the full ancestral reverse process and records raw metrics and run metadata. See [`docs/EVALUATION.md`](docs/EVALUATION.md) for metric definitions and evaluation options.

## Configuration note

The repository retains later experimental configurations for research traceability. They should not be confused with the manuscript formulation:

| Configuration | Prediction | Reverse variance | Epochs | Role |
|---|---|---|---:|---|
| `paper_manuscript_controlled_2048.json` | epsilon | fixed beta | 400 | Manuscript-aligned controlled setup |
| `v035_stable_vpred_2048.json` | v-prediction | posterior | 1,500 | Later stability experiment |

Use the first configuration when following the supplied manuscript. Results from the v0.3.5 configuration are from a different experimental formulation. Additional implementation and reproducibility notes are available in [`docs/`](docs/).

## Citation

Citation details will be added when the paper's author list and publication information are finalized.

## License

A license has not yet been selected. Until a license is added, public visibility of this repository does not grant permission to reuse, modify, or redistribute its contents.
