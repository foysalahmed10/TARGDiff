<div align="center">

# TARGDiff

## Time-Adaptive Routing-Gated Diffusion for 3D Point Cloud Completion

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](#installation)
[![Task](https://img.shields.io/badge/Task-3D%20Point%20Cloud%20Completion-6f42c1)](#overview)
[![Diffusion](https://img.shields.io/badge/Model-Conditional%20Diffusion-0ea5e9)](#method)
[![License](https://img.shields.io/badge/License-Pending-lightgrey)](#license)

**Research code for the paper _TARGDiff: Time-Adaptive Routing-Gated Diffusion for 3D Point Cloud Completion_.**

**Authors:** to be added before public paper release  
**Paper / preprint:** to be added when public  
**Pretrained checkpoints:** to be added after final verification

</div>

---

<p align="center">
  <img src="assets/fig1_overview.png" alt="TARGDiff overview and TARGFuse routing" width="100%">
</p>

<p align="center"><b>Figure 1.</b> TARGDiff uses a dense partial observation <i>P</i> and an observation-derived FPS-256 structure view <i>S</i>. TARGFuse performs token-wise, time-aware routing between independently query-aligned partial and structure contexts, followed by disagreement-aware residual refinement.</p>

> [!IMPORTANT]
> **Scientific scope of this repository.** The supplied manuscript and the maintained codebase contain both the paper formulation and later experimental diffusion variants. The manuscript formulation is represented by `configs/paper_manuscript_controlled_2048.json`. The later `v0.3.5` stability branch uses v-prediction and posterior variance and must not be confused with the paper formulation. Manuscript table values are reproduced below exactly as reported in the supplied paper, but the exact paper checkpoints and audited source split manifest are not bundled in this release; therefore those numerical values should be treated as **paper-reported**, not independently re-verified by this repository snapshot.

## Table of contents

- [Overview](#overview)
- [Highlights](#highlights)
- [Method](#method)
- [Controlled missingness protocol](#controlled-missingness-protocol)
- [Results reported in the manuscript](#results-reported-in-the-manuscript)
- [Qualitative results](#qualitative-results)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Paper formulation vs. later v0.3.5 branch](#paper-formulation-vs-later-v035-branch)
- [Oracle structural-condition ablation](#oracle-structural-condition-ablation)
- [Reproducibility notes](#reproducibility-notes)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)

---

## Overview

Point cloud completion reconstructs a complete 3D shape from an incomplete observation. The problem becomes increasingly under-constrained as visible geometry decreases: a model must preserve observed points while inferring missing parts from limited evidence.

TARGDiff approaches this problem with **conditional point-space diffusion** and a same-source dual-view conditioning design:

- **Dense observation:** the partial cloud `P` preserves the original visible sampling pattern.
- **Coverage-oriented structure:** `S = FPS_256(P)` is a 256-point farthest-point-sampled view of the **same partial observation**.
- **Separate local streams:** local partial and structure features remain separate until query-aligned cross-attention.
- **Shared global conditioning:** global descriptors from both views jointly condition the denoiser.
- **Time-adaptive routing:** TARGFuse decides, for every output token and diffusion step, how much to use each local context.
- **Disagreement refinement:** `|C_p - C_s|` is introduced **after** routing, not as a direct gate input.

The deployable method never uses complete-target geometry to construct the structural condition. The standard 2,048-point controlled model contains **2,182,661 trainable parameters**.

---

## Highlights

The supplied manuscript makes three main contributions:

1. **TARGFuse: time-aware token-wise routing.**  
   For each reverse-diffusion step, the current query attends separately to the dense partial and FPS structure branches. A two-way routing gate uses the query, both contexts, and explicit diffusion time.

2. **Same-source dual conditioning.**  
   The structural branch is not a target-derived prior: it is `FPS_256(P)`, sampled from the same observed cloud. The dense stream emphasizes local sampling detail, while FPS emphasizes wider coverage of visible support.

3. **Controlled missingness evaluation.**  
   The manuscript evaluates separately trained models at **25%, 50%, and 75% missing geometry**, together with Completion3D and ShapeNet-55 benchmarks and mechanism-level ablations.

---

## Method

### 1. Inputs and structural condition

For the controlled experiments:

- partial cloud: `P ∈ R^(B × 2048 × 3)`;
- structure cloud: `S = FPS_256(P) ∈ R^(B × 256 × 3)`;
- clean target: `X0 ∈ R^(B × 2048 × 3)` during training only;
- predicted complete cloud: `X_hat ∈ R^(B × 2048 × 3)`.

`P` and `S` are encoded independently. Their global descriptors are combined early for global conditioning, while local features remain separate so their relative contribution can be reconsidered during denoising.

### 2. Query-aligned dual-view contexts

The denoising backbone produces a query sequence `Q`. Each condition branch is reduced to 64 tokens and aligned to the same query through independently parameterized cross-attention:

```text
C_p = CA_p(Q, T_p)
C_s = CA_s(Q, T_s)
```

where `C_p` and `C_s` are partial-branch and structure-branch contexts defined for the same output tokens.

### 3. TARGFuse routing

TARGFuse projects the diffusion timestep into a token `t_tok` and forms the gate input:

```text
G_in = concat(Q, C_p, C_s, t_tok)
```

A two-way softmax produces per-token routing weights:

```text
[w_p, w_s] = softmax(MLP_g(G_in))
w_p + w_s = 1
```

The routed condition is:

```text
C_f = w_p ⊙ C_p + w_s ⊙ C_s
```

The routing weights represent **relative branch preference**, not calibrated uncertainty.

### 4. Post-routing disagreement refinement

Weighted fusion can hide whether the two branches agree. TARGDiff therefore preserves the symmetric discrepancy:

```text
D = |C_p - C_s|
```

and injects it only after routing:

```text
Q_f = Q + φ(concat(C_f, D))
```

This keeps **allocation** and **disagreement refinement** conceptually separate.

### 5. Training objective

The manuscript uses:

```text
L_main = L_noise + 2 L_CD + L_partial + 0.75 L_struct

L_aux  = 0.25 L_rep
       + 0.10 L_surface
       + 0.02 L_attn
       + 0.01 L_gate

L_total = L_main + L_aux
```

Key definitions:

- `L_noise`: diffusion noise-prediction MSE in the manuscript formulation;
- `L_CD`: symmetric **unsquared Euclidean** Chamfer reconstruction loss;
- `L_partial`: one-sided observed-point consistency (`P → X_hat0`);
- `L_struct`: multi-scale structural consistency using nominal 128/256/512 subsets, clipped to availability;
- `L_rep`: point repulsion regularization;
- `L_surface`: local-spacing regularizer; it does **not** estimate normals or curvature;
- `L_attn`: attention-entropy regularization;
- `L_gate`: weak global gate-balance regularization.

### 6. Reverse diffusion

The paper formulation uses the full **400-step ancestral reverse trajectory**, with a linear beta schedule from `1e-4` to `0.02` and fixed-beta reverse variance. Reduced-step/DDIM evaluation is not part of the maintained paper path.

<p align="center">
  <img src="assets/fig2_architecture.png" alt="Detailed TARGDiff architecture and training flow" width="100%">
</p>

<p align="center"><b>Figure 2.</b> Detailed architecture and training flow. Dashed paths indicate training-only supervision. The complete target is not an inference-time condition.</p>

> **Terminology note.** The detailed diagram contains legacy visual labels such as “reliability gate.” In the manuscript interpretation used here, the two softmax outputs are **relative routing/allocation weights**, not calibrated reliability or uncertainty estimates.

---

## Controlled missingness protocol

The controlled experiment is derived from complete ShapeNet objects distributed with PCN. The supplied manuscript specifies a fixed, disjoint source split of **1,000 training and 200 test shapes across eight categories** before any crop generation.

For each source cloud:

1. center the cloud and scale it to the unit sphere;
2. sample a 2,048-point target with FPS;
3. sample a random unit direction `v`;
4. score points with `s_i = p_i^T v`;
5. for missing ratio `r ∈ {0.25, 0.50, 0.75}`, keep the largest `round((1-r)N)` scores;
6. resample the visible set to a 2,048-point partial cloud `P`;
7. derive `S = FPS_256(P)`;
8. train a **separate model for each missingness level**.

The crop is directional and coherent rather than independent random point dropout. Crop directions are sampled independently for each severity, so the 25%, 50%, and 75% observations are not nested versions of the same missing region.

<p align="center">
  <img src="assets/fig3_protocol.png" alt="Controlled missingness protocol examples" width="96%">
</p>

<p align="center"><b>Figure 3.</b> Controlled missingness examples across Watercraft, Lamp, Airplane, Sofa, Chair, Table, Car, and Cabinet.</p>

---

## Results reported in the manuscript

> [!NOTE]
> The tables below reproduce the values in the supplied manuscript. They are **paper-reported results**. This repository snapshot does not include the exact paper checkpoints and audited split manifest needed to independently establish numerical reproduction.

### Controlled PCN-derived protocol

Models are trained separately for each missingness level. CD-L2 is symmetric squared-Euclidean Chamfer distance and is reported as `×10^4`; lower is better. F@0.02 is higher-is-better.

| Method | 25% CD-L2 ↓ | 25% F@0.02 ↑ | 50% CD-L2 ↓ | 50% F@0.02 ↑ | 75% CD-L2 ↓ | 75% F@0.02 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| PoinTr | 14.81 | 0.842 | 18.44 | 0.650 | 28.43 | 0.423 |
| CRA-PCN | 12.14 | 0.853 | 16.81 | 0.670 | 25.31 | 0.442 |
| SymmCompletion | 13.82 | 0.861 | 15.94 | 0.690 | 23.13 | 0.474 |
| DiffComplete | 11.34 | 0.864 | 15.63 | 0.660 | 24.04 | 0.453 |
| SuperPC | 10.41 | 0.874 | 13.43 | 0.720 | 20.24 | 0.512 |
| 3DDM | 9.54 | 0.883 | 13.70 | 0.720 | 20.83 | 0.504 |
| Simba | 9.03 | 0.882 | 13.12 | 0.730 | 19.71 | 0.521 |
| **TARGDiff (ours)** | **8.71** | **0.896** | **12.43** | **0.760** | **17.34** | **0.573** |

The manuscript reports relative CD-L2 reductions of **3.5%**, **5.3%**, and **12.0%** over the strongest listed CD baseline at 25%, 50%, and 75% missingness, respectively.

### Completion3D

Official Completion3D CD (`10^-4`), lower is better. The benchmark-wide Avg. is the official aggregate score, not the unweighted mean of the eight displayed category columns.

| Method | Avg. | Airplane | Cabinet | Car | Chair | Lamp | Sofa | Table | Watercraft |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PointAttN | **6.63** | 3.28 | 10.77 | 6.13 | 7.14 | **5.92** | 9.72 | 6.16 | 3.59 |
| ASFM-Net | 6.68 | **2.38** | 9.68 | **5.84** | 7.47 | 7.11 | 9.65 | 6.25 | 4.84 |
| SnowflakeNet | 7.60 | 3.48 | 11.09 | 6.90 | 8.75 | 8.42 | 10.15 | 6.46 | 5.32 |
| VE-PCN | 8.10 | 3.83 | 12.74 | 7.86 | 8.66 | 7.24 | 11.47 | 7.88 | 4.75 |
| VRCNet | 8.12 | 3.94 | 10.93 | 6.44 | 9.32 | 8.32 | 11.35 | 8.60 | 5.78 |
| SCRN | 9.13 | 3.35 | 12.81 | 7.78 | 9.88 | 10.12 | 12.95 | 9.77 | 6.10 |
| CRN | 9.21 | 3.38 | 13.17 | 8.31 | 10.62 | 10.00 | 12.86 | 9.16 | 5.80 |
| PMP-Net | 9.23 | 3.99 | 14.70 | 8.55 | 10.21 | 9.27 | 12.43 | 8.51 | 5.77 |
| GRNet | 10.64 | 6.13 | 16.90 | 8.27 | 12.23 | 10.22 | 14.93 | 10.08 | 5.86 |
| SA-Net | 11.22 | 5.27 | 14.45 | 7.78 | 13.67 | 13.53 | 14.22 | 11.75 | 8.84 |
| TopNet | 14.25 | 7.32 | 18.77 | 12.88 | 19.82 | 14.60 | 16.29 | 14.89 | 8.82 |
| PCN | 18.22 | 9.79 | 22.70 | 12.43 | 25.14 | 22.72 | 20.26 | 20.27 | 11.73 |
| FoldingNet | 19.07 | 12.83 | 23.01 | 14.88 | 25.69 | 21.79 | 21.31 | 20.71 | 11.51 |
| **TARGDiff (ours)** | 7.14 | 3.72 | **9.04** | 6.18 | **6.71** | 5.93 | **9.34** | **5.97** | **3.43** |

The manuscript reports that TARGDiff is third-lowest in displayed aggregate CD and lowest in five displayed categories: Cabinet, Chair, Sofa, Table, and Watercraft.

### ShapeNet-55

CD-S, CD-M, and CD-H are CD-L2 (`×10^3`) for Simple, Moderate, and Hard settings. Lower CD and higher F@1% are better.

| Method | CD-S ↓ | CD-M ↓ | CD-H ↓ | CD-Avg ↓ | F@1% ↑ |
|---|---:|---:|---:|---:|---:|
| PCN | 1.94 | 1.96 | 4.08 | 2.66 | 0.133 |
| SnowflakeNet | 0.70 | 1.06 | 1.96 | 1.24 | 0.398 |
| PoinTr | 0.58 | 0.88 | 1.79 | 1.09 | 0.464 |
| AdaPoinTr | 0.49 | 0.69 | 1.24 | 0.81 | 0.503 |
| SymmCompletion | **0.34** | **0.54** | 1.18 | **0.69** | **0.538** |
| **TARGDiff (ours)** | 0.43 | 0.62 | **1.08** | 0.71 | 0.525 |

TARGDiff does not lead every ShapeNet-55 metric; the manuscript reports the lowest listed CD under the **Hard** setting.

### TARGFuse ablation at 50% missingness

| Variant | Struct. | Dual CA | Gate input | Time signal | Disagreement | λ_gate | CD-L2 ×10^4 ↓ | F@0.02 ↑ |
|---|---|---|---|---|---|---:|---:|---:|
| A. Partial only | No | No | — | — | No | — | 14.5 | 0.692 |
| B. Equal dual fusion | Yes | Yes | fixed 0.5/0.5 | None | No | — | 13.7 | 0.708 |
| C. Context gate | Yes | Yes | `[C_p, C_s]` | None | No | 0.01 | 13.1 | 0.719 |
| D. State + zero time | Yes | Yes | `[Q, C_p, C_s, 0]` | Zero token | No | 0.01 | 13.0 | 0.726 |
| E. State + shuffled time | Yes | Yes | `[Q, C_p, C_s, t_tilde]` | Shuffled | No | 0.01 | 13.2 | 0.721 |
| F. State + correct time | Yes | Yes | `[Q, C_p, C_s, t_tok]` | Correct | No | 0.01 | 12.8 | 0.744 |
| **G. Full TARGFuse** | **Yes** | **Yes** | **`[Q, C_p, C_s, t_tok]`** | **Correct** | **Yes** | **0.01** | **12.4** | **0.760** |
| H. Full, no gate reg. | Yes | Yes | `[Q, C_p, C_s, t_tok]` | Correct | Yes | 0 | 12.6 | 0.751 |

### Structural-condition ablation at 50% missingness

| Structural condition | Points | Uses target? | CD-L2 ×10^4 ↓ | F@0.02 ↑ |
|---|---:|---|---:|---:|
| None | 0 | No | 14.5 | 0.692 |
| Random-256(`P`) | 256 | No | 13.9 | 0.728 |
| **FPS-256(`P`)** | **256** | **No** | **12.4** | **0.760** |
| FPS-256(`X0`) **oracle** | 256 | **Yes** | 11.2 | 0.802 |

The complete-target FPS condition is a privileged reference only. Its reported result quantifies structural headroom and is not a deployable setting.

---

## Qualitative results

### 50% missing geometry

<p align="center">
  <img src="assets/fig5_qualitative_50.png" alt="Qualitative comparison at 50 percent missing geometry" width="100%">
</p>

<p align="center"><b>Figure 5.</b> Qualitative comparison at 50% missing geometry. Rows: input, SuperPC, 3DDM, Simba, TARGDiff with TARGFuse, and ground truth.</p>

<details>
<summary><b>Show qualitative results at 25% missing geometry</b></summary>
<br>
<p align="center">
  <img src="assets/fig4_qualitative_25.png" alt="Qualitative comparison at 25 percent missing geometry" width="100%">
</p>
</details>

<details>
<summary><b>Show qualitative results at 75% missing geometry</b></summary>
<br>
<p align="center">
  <img src="assets/fig6_qualitative_75.png" alt="Qualitative comparison at 75 percent missing geometry" width="100%">
</p>
</details>

---

## Repository structure

```text
TARGDiff/
├── README.md
├── GITHUB_SETUP.md
├── CONTRIBUTING.md
├── CITATION.cff.template
├── LICENSE_NOT_SELECTED.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── environment-reference.yml
│
├── assets/                         # Paper figures used by this README
│   ├── fig1_overview.png
│   ├── fig2_architecture.png
│   ├── fig3_protocol.png
│   ├── fig4_qualitative_25.png
│   ├── fig5_qualitative_50.png
│   └── fig6_qualitative_75.png
│
├── configs/
│   ├── paper_manuscript_controlled_2048.json  # Paper-text-aligned controlled config
│   ├── paper_controlled_2048.json             # Historical controlled config
│   ├── legacy_v020_2048.json                  # Historical epsilon/fixed-beta control
│   ├── v035_stable_vpred_2048.json            # Later v-pred/posterior experiment
│   └── ...
│
├── data/
│   └── README.md                    # Dataset layout and preparation guide
│
├── scripts/
│   ├── make_split_manifest.py
│   ├── prepare_dataset.py
│   ├── validate_dataset.py
│   ├── dataset_report.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   └── diagnose_timesteps.py
│
├── src/targdiff/
│   ├── config.py
│   ├── diffusion.py
│   ├── engine.py
│   ├── evaluation.py
│   ├── losses.py
│   ├── metrics.py
│   ├── models/
│   ├── data/
│   └── utils/
│
├── tests/                           # Automated regression/unit tests
├── docs/                            # Scientific/reproducibility notes
└── legacy/                          # Original research scripts retained for provenance
```

---

## Installation

### Recommended

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/TARGDiff.git
cd TARGDiff

python -m pip install -e .
```

For development/tests:

```bash
python -m pip install -r requirements-dev.txt
```

For direct source-tree execution without installation:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

### Environment used in the supplied manuscript

The manuscript reports experiments on:

```text
Ubuntu 20.04.6
Intel Xeon Silver 4210R
124 GiB RAM
2 × NVIDIA RTX 3090 (24 GB)
PyTorch 2.4.1
CUDA 11.8
```

### Additional verified development environment

The maintained v0.3.5 code was also exercised on:

```text
Python 3.10.14
PyTorch 2.5.1
CUDA build 11.8
NumPy 2.2.5
cuDNN 9.1
NVIDIA RTX 2080
```

The package metadata declares broader minimum versions, but exact numerical equivalence across all CUDA/PyTorch combinations is not guaranteed.

Verify your runtime:

```bash
python - <<'PY'
import sys
import numpy as np
import torch

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

---

## Data preparation

This repository does not redistribute PCN, Completion3D, ShapeNet-55, or ShapeNet data.

For the controlled experiment, the **source-object split must be created before directional crop generation** so multiple views of the same object cannot leak across partitions.

Detailed instructions: [`data/README.md`](data/README.md)

### Prepare the main controlled dataset

Assuming an audited split manifest exists at `data/splits/controlled_split.csv`:

```bash
python scripts/prepare_dataset.py \
  --split-manifest data/splits/controlled_split.csv \
  --output-root data/prepared \
  --remove-ratios 0.25 0.50 0.75 \
  --full-points 2048 \
  --partial-points 2048 \
  --structure-points 256 \
  --views-per-ratio 1 \
  --sampler fps \
  --structure-source partial \
  --seed 42
```

The main method must print:

```text
Main-method condition: S = FPS(P).
```

### Validate prepared data

Example for the 50% train split:

```bash
python scripts/validate_dataset.py \
  --data-root data/prepared/remove_50/train \
  --num-partial 2048 \
  --num-structure 256 \
  --num-full 2048 \
  --manifest data/prepared/manifest.csv \
  --require-main-method \
  --require-structure-subset \
  --output-json data/prepared/validate_remove50_train.json
```

> [!WARNING]
> If you regenerate a different source split, your experiment is not an exact reproduction of the manuscript even if every other hyperparameter matches.

---

## Training

### Manuscript-aligned controlled configuration

Use:

```text
configs/paper_manuscript_controlled_2048.json
```

It explicitly encodes the manuscript description:

| Setting | Value |
|---|---|
| Partial points | 2048 |
| Structure points | 256 |
| Output points | 2048 |
| Diffusion steps | 400 |
| Beta schedule | linear, `1e-4 → 0.02` |
| Prediction target | epsilon |
| Reverse variance | fixed beta |
| Batch size | 8 |
| Epochs | 400 |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Min LR | `1e-6` |
| Gradient clipping | 1.0 |
| Seed | 42 |
| EMA | disabled |

Train one separate model per severity:

```bash
# 25% missing
python scripts/train.py \
  --config configs/paper_manuscript_controlled_2048.json \
  --train-root data/prepared/remove_25/train \
  --output runs/paper/remove_25 \
  --device cuda:0

# 50% missing
python scripts/train.py \
  --config configs/paper_manuscript_controlled_2048.json \
  --train-root data/prepared/remove_50/train \
  --output runs/paper/remove_50 \
  --device cuda:0

# 75% missing
python scripts/train.py \
  --config configs/paper_manuscript_controlled_2048.json \
  --train-root data/prepared/remove_75/train \
  --output runs/paper/remove_75 \
  --device cuda:0
```

If a validation partition is available, add:

```bash
--val-root data/prepared/remove_50/val
```

When no validation loader is supplied, `best.pt` falls back to **training-loss selection**. Do not describe that checkpoint as validation-selected. For a no-validation paper run, a predefined final-epoch checkpoint is easier to interpret than selecting by test-set performance.

For more detail, see [`docs/PAPER_REPRODUCTION.md`](docs/PAPER_REPRODUCTION.md).

---

## Evaluation

The maintained evaluator uses the **full ancestral reverse chain**.

### Controlled-study metrics

- **CD-L2:** symmetric squared-Euclidean Chamfer distance;
- display scale: `×10^4`;
- **F-score:** threshold `0.02`;
- training Chamfer and evaluation CD-L2 are intentionally different definitions.

Example:

```bash
python scripts/evaluate.py \
  --checkpoint runs/paper/remove_50/checkpoints/epoch_0400.pt \
  --data-root data/prepared/remove_50/test \
  --output evaluations/paper/remove_50 \
  --device cuda:0 \
  --weights raw \
  --seed 42 \
  --repeats 1 \
  --f-threshold 0.02 \
  --cd-scale 10000
```

The output `summary.json` records formulation metadata including prediction type, schedule type, variance type, terminal SNR, and actual weight set used.

For paper-quality reporting, specify `--weights`, `--seed`, `--repeats`, `--f-threshold`, and `--cd-scale` explicitly rather than relying on implicit defaults.

> [!NOTE]
> The maintained controlled-data pipeline is implemented in this repository. Official Completion3D and ShapeNet-55 data are not redistributed here, and dataset-specific benchmark adapters should only be added from verified official structures/protocols. The benchmark tables above reproduce the supplied manuscript; they are not claimed to be re-generated from this repository snapshot alone.

---

## Inference

Inference requires only a partial point cloud. The structure condition is generated internally from the observation as `S = FPS(P)`.

```bash
python scripts/infer.py \
  --checkpoint /path/to/checkpoint.pt \
  --partial /path/to/partial_cloud.txt \
  --output outputs/completed_cloud.txt \
  --device cuda:0 \
  --weights raw \
  --seed 42
```

Use `--normalize-partial` only if that matches the preprocessing used for the checkpoint.

---

## Paper formulation vs. later v0.3.5 branch

The repository intentionally preserves multiple diffusion formulations for research provenance.

| Config | Purpose | Prediction | Schedule | Reverse variance | Epochs |
|---|---|---|---|---|---:|
| `paper_manuscript_controlled_2048.json` | **Supplied manuscript formulation** | epsilon | linear | fixed beta | 400 |
| `legacy_v020_2048.json` | Historical control | epsilon | linear | fixed beta | historical |
| `v035_stable_vpred_2048.json` | Later stability experiment | **v-prediction** | linear | **posterior** | 1500 |
| `v030_recommended_2048.json` | Earlier experimental branch | v-prediction | zero-terminal-SNR rescaling | posterior | experimental |

The manuscript describes noise prediction and fixed-beta ancestral variance. Therefore results from `v035_stable_vpred_2048.json` must be labeled as a separate experimental branch rather than attributed to the paper configuration.

---

## Oracle structural-condition ablation

The deployable method is:

```text
S = FPS_256(P)
```

For the structural ablation only, the repository can generate:

```text
S_oracle = FPS_256(X0)
```

This uses complete-target information and is unavailable at inference.

The preparer therefore requires an explicit flag:

```bash
python scripts/prepare_dataset.py \
  --split-manifest data/splits/controlled_split.csv \
  --output-root data/prepared_oracle \
  --remove-ratios 0.50 \
  --full-points 2048 \
  --partial-points 2048 \
  --structure-points 256 \
  --sampler fps \
  --structure-source full \
  --allow-privileged-oracle \
  --seed 42
```

Any result from this setting must be labeled **oracle / privileged / non-deployable**.

---

## Reproducibility notes

For every reported experiment, archive together:

- code commit hash;
- resolved configuration;
- source split manifest and its SHA-256;
- dataset metadata;
- checkpoint;
- checkpoint-selection rule;
- evaluation command;
- evaluation seed and repeat count;
- raw vs. EMA weight selection;
- metric thresholds/scales;
- per-sample outputs when possible.

Useful repository documents:

- [`docs/PAPER_REPRODUCTION.md`](docs/PAPER_REPRODUCTION.md)
- [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md)
- [`docs/EVALUATION.md`](docs/EVALUATION.md)
- [`docs/SCIENTIFIC_ALIGNMENT.md`](docs/SCIENTIFIC_ALIGNMENT.md)
- [`docs/PUBLIC_RELEASE_AUDIT.md`](docs/PUBLIC_RELEASE_AUDIT.md)

Run the automated tests with:

```bash
pytest
```

and compile-check Python sources with:

```bash
python -m compileall -q src scripts
```

---

## Limitations

The supplied manuscript identifies several limitations:

- the 64 condition tokens are selected by point-array order rather than a permutation-invariant geometric rule;
- structural-loss subsets are also order-dependent;
- controlled directional crops do not reproduce sensor noise, non-uniform density, or complex real-world occlusion;
- routing-weight evolution across spatial locations and diffusion timesteps is not directly analyzed;
- the controlled 25%, 50%, and 75% settings use **separately trained models**, so they do not establish cross-severity robustness of one model;
- Completion3D, ShapeNet-55, and the controlled PCN-derived experiment are complementary but distinct protocols.

These limitations should be preserved when interpreting or extending the results.

---

## Citation

Author metadata is intentionally left as a placeholder in this release. Before making the repository public, complete `CITATION.cff.template` and rename it to `CITATION.cff`.

BibTeX template:

```bibtex
@article{targdiff2026,
  title   = {TARGDiff: Time-Adaptive Routing-Gated Diffusion for 3D Point Cloud Completion},
  author  = {<AUTHOR LIST>},
  journal = {<VENUE OR PREPRINT>},
  year    = {2026},
  url     = {<PAPER OR PROJECT URL>}
}
```

---

## License

No open-source license has been selected in this repository snapshot. See [`LICENSE_NOT_SELECTED.md`](LICENSE_NOT_SELECTED.md).

Before public release, the project owners should select a license only after confirming institutional rights, dataset restrictions, dependency obligations, and provenance of any retained legacy code.

---

## Acknowledgment of experimental scope

TARGDiff is research software. Benchmark numbers depend on exact preprocessing, source splits, diffusion formulation, checkpoint provenance, and evaluation settings. Please report the configuration and protocol used when publishing results derived from this repository.
