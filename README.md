<div align="center">

# TARGDiff

### Time-Adaptive Routing-Gated Diffusion for 3D Point Cloud Completion

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Task](https://img.shields.io/badge/Task-3D%20Point%20Cloud%20Completion-6f42c1)](#overview)
[![Model](https://img.shields.io/badge/Model-Conditional%20Diffusion-0ea5e9)](#method)
[![License](https://img.shields.io/badge/License-Not%20Selected-lightgrey)](#license)

**Research implementation of TARGDiff for conditional 3D point-cloud completion.**

**Authors:** Final author list to be added after approval  
**Paper:** Publication or preprint details to be added when public  
**Pretrained checkpoints:** Not included in this source release

</div>

<p align="center">
  <img src="assets/fig1_overview.png" alt="Figure 1. TARGDiff overview and TARGFuse routing" width="100%">
</p>

<p align="center"><strong>Figure 1.</strong> TARGDiff overview. The dense partial observation and its FPS-based coverage view are encoded separately. TARGFuse aligns both local streams to the current denoising query, performs time-adaptive token-wise routing, and applies disagreement-aware refinement after routing.</p>

---

> [!IMPORTANT]
> **Scientific and release scope.** This repository contains the TARGDiff research implementation, the manuscript-aligned controlled configuration, and later experimental diffusion branches. The manuscript formulation is represented by `configs/paper_manuscript_controlled_2048.json`. The later v0.3.5 stability branch uses v-prediction and posterior reverse variance; it is a separate experimental formulation and must not be reported as the manuscript configuration.
>
> The benchmark values in this README are transcribed from the supplied manuscript. The exact paper checkpoints and audited source-object split manifest are not included in this source release. These values are therefore identified as **manuscript-reported results**, not as results independently reproduced from this ZIP.

## Table of contents

- [Overview](#overview)
- [Languages and technologies](#languages-and-technologies)
- [Highlights](#highlights)
- [Method](#method)
- [Controlled missingness protocol](#controlled-missingness-protocol)
- [Experimental results](#experimental-results)
  - [Controlled 25%–75% missingness](#controlled-25--75-missingness)
  - [Completion3D](#completion3d)
  - [ShapeNet-55](#shapenet-55)
  - [TARGFuse ablation](#targfuse-ablation-at-50-missingness)
  - [Structural-condition ablation](#structural-condition-ablation-at-50-missingness)
  - [Sampling efficiency](#sampling-efficiency-reported-in-the-manuscript)
- [All paper figures](#all-paper-figures)
- [Repository structure](#repository-structure)
- [Environment and installation](#environment-and-installation)
- [Dataset preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Diagnostics and tests](#diagnostics-and-tests)
- [Manuscript configuration and experimental branches](#manuscript-configuration-and-experimental-branches)
- [Oracle structural-condition experiment](#oracle-structural-condition-experiment)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [License](#license)
- [Citation](#citation)

---

## Overview

3D point-cloud completion aims to reconstruct a complete object from an incomplete geometric observation. As more geometry is missing, the task becomes increasingly under-constrained: a model must preserve measured evidence while inferring the unseen parts of the object.

**TARGDiff** is a conditional point-space diffusion framework that studies how complementary representations of the same partial observation should be used during the reverse-diffusion process.

The method uses two observation-derived views:

- **Dense partial observation (`P`):** preserves the original visible sampling pattern and local geometric detail.
- **FPS coverage view (`S`):** a compact 256-point farthest-point-sampled representation derived from the same partial observation. It spreads the available point budget more broadly across the visible shape.

The two views are encoded independently. Their global descriptors jointly condition the denoising backbone, while their local features remain separate until they are independently aligned to the current denoising query.

The central module, **TARGFuse**, predicts token-wise routing weights from the current query, the two query-aligned contexts, and the diffusion timestep. The explicitly computed disagreement between the two contexts is used after routing for residual refinement.

**The complete target is used for training supervision and evaluation only. It is not an inference-time conditioning input.** The deployable structural view is always derived from the partial observation.

## Languages and technologies

The repository is Python-first. GitHub may display its own language percentages based on the files in the public repository.

| Language or format | Role in this repository |
|---|---|
| **Python** | Core model, diffusion process, point-cloud sampling, data preparation, training, evaluation, inference, and tests |
| **Markdown** | Main README, scientific documentation, release notes, and usage guides |
| **JSON** | Model, loss, and training configurations |
| **YAML** | Reference Conda environment |
| **TOML** | Python package metadata and build configuration |
| **TXT** | Dependency lists and supported point-cloud text data |

**Core framework:** PyTorch  
**Numerical processing:** NumPy  
**Testing:** pytest  
**Packaging:** setuptools with `pyproject.toml`

The current source release does not include a custom C++ or CUDA extension; the maintained model and geometry operations are implemented in Python/PyTorch.

## Highlights

1. **Same-source dual-view conditioning**  
   The dense and FPS branches are derived from the same visible partial cloud. The structural branch does not introduce complete-target geometry into the deployable inference path.

2. **Independent local encoding and query alignment**  
   The two condition branches retain separate local features and independently answer the same evolving denoising query through cross-attention.

3. **Time-adaptive token-wise routing**  
   TARGFuse predicts a pair of relative branch-allocation weights for every output token and denoising step.

4. **Post-routing disagreement refinement**  
   The explicit difference between the two aligned contexts is introduced in the residual-refinement pathway after the routing decision.

5. **Controlled missingness evaluation**  
   The manuscript evaluates separately trained models at 25%, 50%, and 75% missing geometry, alongside Completion3D, ShapeNet-55, component ablations, and a sampling-efficiency study.

6. **Research and reproducibility tools**  
   The repository includes source-level split creation, dataset preparation and validation, training/evaluation launchers, timestep diagnostics, experiment configurations, tests, and release documentation.

## Method

### 1. Dual-view observation conditioning

The input is represented in two complementary ways:

- The dense branch retains the local sampling characteristics of the partial point cloud.
- The FPS branch samples 256 points from that same partial observation to provide a more spatially distributed view of the visible geometry.

Both branches are encoded independently. Their global descriptors are combined to provide object-level conditioning to the denoising backbone. Their local features remain separate so that the model can choose their relative contribution later.

### 2. Query-aligned dual contexts

The denoising backbone produces a query representation for the current noisy point cloud. Each condition branch is reduced to 64 local tokens and independently processed by its own four-head cross-attention module.

This produces two contexts aligned to the same output queries:

- **Partial context:** local evidence from the dense observation.
- **Structure context:** coverage-oriented evidence from the FPS view.

The two contexts therefore describe the same output locations, but they are derived from different samplings of the observed geometry.

### 3. TARGFuse: time-adaptive routing

TARGFuse receives the current denoising query, both query-aligned contexts, and a projected diffusion-time representation.

A small MLP followed by a two-way softmax predicts the relative contribution of the partial and structure branches for each output token. The two weights are non-negative and sum to one.

These weights represent **learned relative branch preference**. They are not calibrated uncertainty estimates or probabilities that a branch is correct.

### 4. Disagreement-aware residual refinement

A weighted combination alone can hide whether the two contexts agree. TARGDiff therefore retains their symmetric element-wise disagreement as an additional descriptor.

The explicit disagreement descriptor is kept outside the routing input and is introduced only in the post-allocation residual-refinement pathway. This separates two roles:

- routing allocates the relative contribution of the two contexts;
- residual refinement can respond to disagreement between them.

### 5. Diffusion backbone and learning objective

The manuscript-aligned controlled model uses a 400-step linear diffusion schedule and an EdgeConv-based denoising backbone.

The training objective combines:

- diffusion noise-prediction loss;
- symmetric Chamfer reconstruction loss;
- one-sided partial-observation consistency;
- structural consistency at multiple subset sizes;
- point-repulsion regularization;
- a local-spacing regularizer;
- attention-entropy regularization;
- weak global gate-balance regularization.

The local-spacing term is a geometric regularizer. It does not estimate surface normals or curvature.

### 6. Inference

At inference, the model receives a partial point cloud and constructs its FPS coverage view from the observed input. It then runs the complete ancestral reverse-diffusion process for the checkpoint's configured number of steps.

The complete target is not required for inference.

<p align="center">
  <img src="assets/fig2_architecture.png" alt="Figure 2. Detailed TARGDiff architecture and training flow" width="100%">
</p>

<p align="center"><strong>Figure 2.</strong> Detailed TARGDiff architecture and training flow. The dashed supervision paths are training-only; the complete target is not part of the inference-time conditioning pathway.</p>

---

## Controlled missingness protocol

The controlled study is derived from complete ShapeNet objects distributed with PCN. It is designed to make the amount of missing geometry an explicit experimental variable.

The manuscript specifies:

- eight object categories;
- a fixed, disjoint source-object split of 1,000 training and 200 test shapes;
- source-level splitting before crop generation;
- 2,048 points for the complete target and partial observation;
- a 256-point FPS coverage view derived from the partial observation;
- directional cropping at 25%, 50%, and 75% missing geometry;
- a separate model trained for each missingness level.

The directional crop removes a coherent region of the object rather than independently dropping random points. The crop direction is sampled independently for each severity, so the 25%, 50%, and 75% observations are not nested versions of one crop.

<p align="center">
  <img src="assets/fig3_protocol.png" alt="Figure 3. Controlled missingness protocol across eight object categories" width="100%">
</p>

<p align="center"><strong>Figure 3.</strong> Controlled missingness examples. Columns show Watercraft, Lamp, Airplane, Sofa, Chair, Table, Car, and Cabinet. Rows show the complete target and the 25%, 50%, and 75% missing-geometry settings.</p>

> [!WARNING]
> The exact audited source-object split manifest used for the manuscript is not bundled in this release. Creating a new deterministic split is useful for running the pipeline, but it is a new experiment and must not be described as an exact reproduction of the manuscript split.

---

## Experimental results

> [!NOTE]
> All numerical tables in this section reproduce the supplied manuscript. They are **manuscript-reported results**. The exact paper checkpoints and audited source split manifest are not included in this source release, so this repository snapshot alone cannot independently reproduce or verify the reported table entries.

### Controlled 25%–75% missingness

The controlled experiment trains and evaluates a separate model for each missingness level. CD-L2 is symmetric squared-Euclidean Chamfer distance, reported with a scale factor of 10⁴; lower is better. F-score is measured at a distance threshold of 0.02; higher is better.

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

The manuscript reports the following CD-L2 reductions over the strongest listed CD baseline:

| Missing geometry | TARGDiff CD-L2 | Relative reduction |
|---:|---:|---:|
| 25% | 8.71 | 3.5% |
| 50% | 12.43 | 5.3% |
| 75% | 17.34 | 12.0% |

The relative gap over the listed baseline increases with missingness in this controlled protocol. However, the absolute completion quality also decreases as less geometry is observed. Because a separate model is trained at each severity, this experiment does **not** establish cross-severity generalization by one model.

### Completion3D

The following table reproduces the Completion3D results in the manuscript. The benchmark's official CD scale is 10⁻⁴; lower is better. The `Avg.` column is the benchmark-wide official aggregate, not the unweighted mean of the eight displayed category values.

| Method | Avg. | Airplane | Cabinet | Car | Chair | Lamp | Sofa | Table | Watercraft |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PointAttN | **6.63** | 3.28 | 10.77 | 6.13 | 7.14 | 5.92 | 9.72 | 6.16 | 3.59 |
| ASFM-Net | 6.68 | 2.38 | 9.68 | 5.84 | 7.47 | 7.11 | 9.65 | 6.25 | 4.84 |
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
| **TARGDiff (ours)** | **7.14** | 3.72 | **9.04** | 6.18 | **6.71** | 5.93 | **9.34** | **5.97** | **3.43** |

TARGDiff obtains an average CD of 7.14. It is the third-lowest result among the displayed methods with complete category-wise rows, after PointAttN and ASFM-Net. FSC reports an overall value of 6.66 in the manuscript discussion, but its compatible eight-category breakdown is not included in the table. TARGDiff has the lowest displayed CD in five categories: Cabinet, Chair, Sofa, Table, and Watercraft.

### ShapeNet-55

ShapeNet-55 contains 55 categories and reports Simple, Moderate, and Hard incompleteness settings. The complete target and prediction contain 8,192 points, while the partial observation contains 2,048 points. CD-L2 is reported with a scale factor of 10³; lower is better. F-score@1% is higher-is-better.

| Method | CD-S ↓ | CD-M ↓ | CD-H ↓ | CD-Avg ↓ | F@1% ↑ |
|---|---:|---:|---:|---:|---:|
| PCN | 1.94 | 1.96 | 4.08 | 2.66 | 0.133 |
| SnowflakeNet | 0.70 | 1.06 | 1.96 | 1.24 | 0.398 |
| PoinTr | 0.58 | 0.88 | 1.79 | 1.09 | 0.464 |
| AdaPoinTr | 0.49 | 0.69 | 1.24 | 0.81 | 0.503 |
| SymmCompletion | **0.34** | **0.54** | 1.18 | **0.69** | **0.538** |
| **TARGDiff (ours)** | 0.43 | 0.62 | **1.08** | 0.71 | 0.525 |

TARGDiff does not lead every ShapeNet-55 metric. Its strongest listed result is the Hard-setting CD-L2 of 1.08, which is lower than the other displayed methods in that column.

### TARGFuse ablation at 50% missingness

The ablation study examines the structural branch, dual cross-attention, learned routing, direct diffusion-time input, post-routing disagreement refinement, and gate-balance regularization.

| Variant | Structural branch | Dual CA | Gate input | Time signal | Disagreement refinement | λ_gate | CD-L2 ×10⁴ ↓ | F@0.02 ↑ |
|---|---|---|---|---|---|---:|---:|---:|
| A. Partial only | No | No | — | — | No | — | 14.5 | 0.692 |
| B. Equal dual fusion | Yes | Yes | Fixed 0.5 / 0.5 | None | No | — | 13.7 | 0.708 |
| C. Context gate | Yes | Yes | Both contexts | None | No | 0.01 | 13.1 | 0.719 |
| D. State + zero time | Yes | Yes | Query + contexts + zero token | Zero | No | 0.01 | 13.0 | 0.726 |
| E. State + shuffled time | Yes | Yes | Query + contexts + shuffled time | Shuffled | No | 0.01 | 13.2 | 0.721 |
| F. State + correct time | Yes | Yes | Query + contexts + time | Correct | No | 0.01 | 12.8 | 0.744 |
| **G. Full TARGFuse** | **Yes** | **Yes** | **Query + contexts + time** | **Correct** | **Yes** | **0.01** | **12.4** | **0.760** |
| H. Full, no gate regularizer | Yes | Yes | Query + contexts + time | Correct | Yes | 0 | 12.6 | 0.751 |

The reported comparisons support the intended roles of direct time information and post-routing disagreement refinement. The gate-balance regularizer provides a smaller additional improvement; it is not the only source of the routing benefit.

### Structural-condition ablation at 50% missingness

This experiment distinguishes the benefit of adding a second sparse view from the benefit of FPS-based coverage.

| Structural condition | Point count | Complete-target information used? | CD-L2 ×10⁴ ↓ | F@0.02 ↑ |
|---|---:|---|---:|---:|
| None | 0 | No | 14.5 | 0.692 |
| Random-256 from partial input | 256 | No | 13.9 | 0.728 |
| **FPS-256 from partial input** | **256** | **No** | **12.4** | **0.760** |
| FPS-256 from complete target (oracle) | 256 | **Yes** | 11.2 | 0.802 |

The complete-target FPS condition is a privileged oracle reference. It is unavailable at inference and is not the deployable TARGDiff input setting.

### Sampling efficiency reported in the manuscript

The manuscript also reports a quality-latency comparison using the same 50% checkpoint, test set, hardware, and batch size.

| Sampler | Denoiser evaluations | CD-L2 ×10⁴ ↓ | F@0.02 ↑ | Latency per shape ↓ | Speedup ↑ | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| Ancestral | 400 | 12.43 | 0.760 | 1.60 s | 1.00× | 4.8 GB |
| DDIM | 100 | 12.55 | 0.755 | 0.46 s | 3.48× | 4.7 GB |
| DDIM | 50 | 12.80 | 0.747 | 0.27 s | 5.93× | 4.7 GB |
| DDIM | 25 | 13.30 | 0.731 | 0.17 s | 9.41× | 4.6 GB |

These are manuscript-reported measurements on one RTX 3090 at inference batch size 1. **The maintained public evaluation CLI in this release uses the full ancestral reverse chain; it does not expose the reduced-step DDIM schedules shown in this manuscript table.** Do not claim that the current CLI reproduces the DDIM timings.

---

## All paper figures

All six original manuscript figures are included in `assets/` and are displayed below. The image files are retained in the repository so they can also be opened directly from GitHub.

### Figure 1 — TARGDiff overview

<p align="center">
  <img src="assets/fig1_overview.png" alt="Figure 1. TARGDiff overview" width="100%">
</p>

### Figure 2 — Detailed architecture

<p align="center">
  <img src="assets/fig2_architecture.png" alt="Figure 2. TARGDiff detailed architecture" width="100%">
</p>

### Figure 3 — Controlled missingness protocol

<p align="center">
  <img src="assets/fig3_protocol.png" alt="Figure 3. Controlled missingness protocol" width="100%">
</p>

### Figure 4 — Qualitative results at 25% missing geometry

<p align="center">
  <img src="assets/fig4_qualitative_25.png" alt="Figure 4. Qualitative point-cloud completion results at 25 percent missing geometry" width="100%">
</p>

### Figure 5 — Qualitative results at 50% missing geometry

<p align="center">
  <img src="assets/fig5_qualitative_50.png" alt="Figure 5. Qualitative point-cloud completion results at 50 percent missing geometry" width="100%">
</p>

### Figure 6 — Qualitative results at 75% missing geometry

<p align="center">
  <img src="assets/fig6_qualitative_75.png" alt="Figure 6. Qualitative point-cloud completion results at 75 percent missing geometry" width="100%">
</p>

---

## Repository structure

The following tree reflects the source release. Some directories contain additional implementation and test files; the documentation links below describe their roles.

```text
TARGDiff_GitHub_Release/
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── GITHUB_SETUP.md
├── PUBLICATION_RELEASE_CHECKLIST.md
├── CITATION.cff.template
├── LICENSE_NOT_SELECTED.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── environment-reference.yml
│
├── assets/
│   ├── fig1_overview.png
│   ├── fig2_architecture.png
│   ├── fig3_protocol.png
│   ├── fig4_qualitative_25.png
│   ├── fig5_qualitative_50.png
│   └── fig6_qualitative_75.png
│
├── configs/
│   ├── debug_smoke.json
│   ├── legacy_v020_2048.json
│   ├── paper_controlled_2048.json
│   ├── paper_manuscript_controlled_2048.json
│   ├── v030_recommended_2048.json
│   ├── v032_recommended_2048.json
│   ├── v033_recommended_2048.json
│   ├── v034_recommended_2048.json
│   └── v035_stable_vpred_2048.json
│
├── data/
│   └── README.md
│
├── docs/
│   ├── CODE_AUDIT.md
│   ├── DATA_PREPARATION.md
│   ├── EVALUATION.md
│   ├── EXPERIMENTAL_RESULTS_RECORD.md
│   ├── GITHUB_RELEASE_VERIFICATION.md
│   ├── GPU_STABILITY_FINDINGS_V035.md
│   ├── NEXT_RETRAINING_PROTOCOL.md
│   ├── PAPER_REPRODUCTION.md
│   ├── PUBLIC_RELEASE_AUDIT.md
│   ├── REPRODUCIBILITY.md
│   ├── SCIENTIFIC_ALIGNMENT.md
│   ├── VERIFICATION_REPORT.md
│   └── CUDA and version-specific audit notes
│
├── legacy/
│   ├── README.md
│   ├── data_loader_diffusion.py
│   ├── dbdiffusion_model_targfuse.py
│   ├── losses_diffusion_targfuse.py
│   ├── prepare_dbdiffusion_completion_dataset_multiclass.py
│   └── train_dbdiffusion_targfuse.py
│
├── scripts/
│   ├── make_split_manifest.py
│   ├── prepare_dataset.py
│   ├── validate_dataset.py
│   ├── dataset_report.py
│   ├── train.py
│   ├── train_controlled_suite.py
│   ├── evaluate.py
│   ├── evaluate_controlled_suite.py
│   ├── infer.py
│   ├── diagnose_timesteps.py
│   ├── gpu_preflight_worker.py
│   └── gpu_ab_preflight.py
│
├── src/
│   └── targdiff/
│       ├── config.py
│       ├── diffusion.py
│       ├── engine.py
│       ├── evaluation.py
│       ├── losses.py
│       ├── metrics.py
│       ├── training.py
│       ├── validation.py
│       ├── data/
│       │   ├── dataset.py
│       │   ├── io.py
│       │   ├── prepare.py
│       │   └── sampling.py
│       ├── models/
│       │   ├── components.py
│       │   └── targdiff.py
│       └── utils/
│           ├── checkpoint.py
│           ├── ema.py
│           ├── reproducibility.py
│           └── runtime.py
│
└── tests/
    ├── test_model.py
    ├── test_engine.py
    ├── test_losses_metrics.py
    ├── test_prepare_and_dataset.py
    ├── test_validation_determinism.py
    └── additional regression and CLI tests
```

### Important files

| Path | Purpose |
|---|---|
| `src/targdiff/models/targdiff.py` | TARGDiff model and denoising architecture |
| `src/targdiff/models/components.py` | Model building blocks and TARGFuse components |
| `src/targdiff/diffusion.py` | Forward and reverse diffusion operations |
| `src/targdiff/losses.py` | Training losses and geometric regularizers |
| `src/targdiff/data/prepare.py` | Source-level split and controlled dataset preparation |
| `src/targdiff/data/dataset.py` | Prepared point-cloud dataset loading |
| `src/targdiff/runner.py` | Training loop, checkpointing, and history |
| `src/targdiff/evaluation.py` | Evaluation summaries and formulation metadata |
| `scripts/train_controlled_suite.py` | Sequential training for 25%, 50%, and 75% settings |
| `scripts/evaluate_controlled_suite.py` | Evaluation and aggregation for all controlled settings |
| `configs/paper_manuscript_controlled_2048.json` | Manuscript-aligned controlled configuration |
| `environment-reference.yml` | Reference Conda environment |

---

## Environment and installation

### System requirements

| Component | Requirement or reference |
|---|---|
| Operating system | Linux recommended; Ubuntu was used in the reported experiments |
| Python | 3.10 or later |
| PyTorch | 2.0.1 or later according to package metadata |
| NumPy | 1.23.5 or later according to package metadata |
| GPU | NVIDIA GPU recommended for practical training |
| CUDA | Use a PyTorch build compatible with the installed NVIDIA driver |
| Memory | Depends on point count, batch size, and GPU; manuscript experiments used two RTX 3090 GPUs |

The project metadata declares minimum package versions. Those minimums are not a guarantee that every PyTorch/CUDA combination produces identical performance or numerical results.

### Reference environment

The repository includes `environment-reference.yml`, based on the verified development environment:

| Component | Reference version |
|---|---|
| Python | 3.10.14 |
| NumPy | 2.2.5 |
| PyTorch | 2.5.1 |
| PyTorch CUDA runtime | 11.8 |
| cuDNN | 9.1 |
| Development GPU | NVIDIA RTX 2080 |

The manuscript reports a separate experiment environment:

- Ubuntu 20.04.6;
- Intel Xeon Silver 4210R;
- 124 GiB RAM;
- two NVIDIA RTX 3090 GPUs with 24 GB each;
- PyTorch 2.4.1;
- CUDA 11.8.

### Option A — create the reference Conda environment

From the repository root:

```bash
conda env create -f environment-reference.yml
conda activate targdiff-reference

python -m pip install -e ".[dev]"
```

This installs the package in editable mode and includes the development test dependency.

### Option B — use an existing compatible Conda environment

Activate an environment that already has a compatible Python, NumPy, and PyTorch installation:

```bash
conda activate YOUR_EXISTING_ENVIRONMENT

python -m pip install -e ".[dev]"
```

If you use an existing environment, record its Python, PyTorch, CUDA, NumPy, and GPU versions with the verification command below. Avoid changing a validated environment in the middle of a controlled experiment.

### Option C — create a Python virtual environment

Create and activate a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
```

Install a PyTorch build compatible with your operating system and CUDA/CPU setup using the official PyTorch installation selector:

https://pytorch.org/get-started/locally/

Then install TARGDiff:

```bash
python -m pip install -e ".[dev]"
```

### Verify the environment

```bash
python - <<'PY'
import sys
import numpy as np
import torch

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        print(f"GPU {index}:", torch.cuda.get_device_name(index))
PY
```

For an NVIDIA system, also inspect the driver and GPU:

```bash
nvidia-smi
```

### Install without editable packaging

If you prefer not to install the package, run from the repository root and expose the source directory:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

This is also useful when running scripts directly from a source checkout.

---

## Dataset preparation

### Dataset availability

This repository does **not** redistribute PCN, ShapeNet, Completion3D, ShapeNet-55, or other benchmark datasets. Obtain datasets from their official sources and follow their terms of use.

The controlled-data preparation pipeline accepts point clouds with the following extensions:

- `.txt`
- `.xyz`
- `.pts`
- `.npy`

Each input root represents one class. An example layout is:

```text
data/
└── raw/
    ├── Airplane/
    ├── Cabinet/
    ├── Car/
    ├── Chair/
    ├── Lamp/
    ├── Sofa/
    ├── Table/
    └── Watercraft/
```

### Step 1 — create a source-object split

Create the split **before** generating crops. This prevents multiple crops or views of the same source object from leaking between training, validation, and test partitions.

```bash
mkdir -p data/splits

python scripts/make_split_manifest.py \
  --input-roots \
    data/raw/Airplane \
    data/raw/Cabinet \
    data/raw/Car \
    data/raw/Chair \
    data/raw/Lamp \
    data/raw/Sofa \
    data/raw/Table \
    data/raw/Watercraft \
  --output data/splits/controlled_split.csv \
  --train 0.8 \
  --val 0.1 \
  --test 0.1 \
  --seed 42
```

The command above creates a new deterministic, class-stratified split using the requested ratios. It is an example workflow, **not a replacement for the original audited manuscript split**. Use the original split manifest if it becomes available and exact paper reproduction is required.

### Step 2 — prepare all three missingness levels

Use the source-level manifest created in Step 1:

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

For the deployable main method, keep `--structure-source partial`. This ensures that the structure branch is derived from the observed partial cloud.

### Step 3 — validate the prepared dataset

Example for the 50% training split:

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

You can run the same validation for `remove_25/train`, `remove_75/train`, and the corresponding test splits.

### Step 4 — inspect the preparation manifest

```bash
python scripts/dataset_report.py \
  --manifest data/prepared/manifest.csv \
  --output data/prepared/dataset_report.json \
  --require-main-method
```

### Prepared dataset layout

```text
data/prepared/
├── manifest.csv
├── dataset_metadata.json
├── remove_25/
│   ├── train/<class>/
│   │   ├── partial/
│   │   ├── structure/
│   │   └── full/
│   ├── val/<class>/
│   └── test/<class>/
├── remove_50/
│   ├── train/<class>/
│   ├── val/<class>/
│   └── test/<class>/
└── remove_75/
    ├── train/<class>/
    ├── val/<class>/
    └── test/<class>/
```

Each prepared sample has paired partial, structure, and full point clouds. The dataset loader checks the expected point counts and pairs the files by sample key.

See [`data/README.md`](data/README.md) and [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md) for additional details.

---

## Training

### Manuscript-aligned configuration

Use:

```text
configs/paper_manuscript_controlled_2048.json
```

This configuration explicitly matches the manuscript description of the controlled experiment.

| Setting | Value |
|---|---|
| Partial points | 2,048 |
| Structure points | 256 |
| Output points | 2,048 |
| Diffusion steps | 400 |
| Beta schedule | Linear |
| Beta start | 0.0001 |
| Beta end | 0.02 |
| Prediction target | Epsilon |
| Reverse variance | Fixed beta |
| Optimizer | AdamW |
| Batch size | 8 |
| Epochs | 400 |
| Learning rate | 0.0001 |
| Minimum learning rate | 0.000001 |
| Weight decay | 0.0001 |
| Gradient clipping | 1.0 |
| Scheduler | Cosine annealing |
| Random seed | 42 |
| EMA | Disabled |

The manuscript-aligned model contains **2,182,661 trainable parameters**.

### Option A — train the three severities with one command

The controlled-suite launcher trains separate models for the 25%, 50%, and 75% datasets:

```bash
python scripts/train_controlled_suite.py \
  --prepared-root data/prepared \
  --runs-root runs/paper \
  --config configs/paper_manuscript_controlled_2048.json \
  --severities 25 50 75 \
  --device cuda:0
```

To print the commands without starting training, add:

```bash
--dry-run
```

The launcher checks for a validation directory for each severity. If one exists, it passes that directory to the training script.

### Option B — train one severity manually

Example for 50% missing geometry:

```bash
python scripts/train.py \
  --config configs/paper_manuscript_controlled_2048.json \
  --train-root data/prepared/remove_50/train \
  --val-root data/prepared/remove_50/val \
  --output runs/paper/remove_50 \
  --device cuda:0
```

If no validation split exists, omit `--val-root`.

The corresponding commands for the other severities are:

```bash
# 25% missing geometry
python scripts/train.py \
  --config configs/paper_manuscript_controlled_2048.json \
  --train-root data/prepared/remove_25/train \
  --output runs/paper/remove_25 \
  --device cuda:0

# 75% missing geometry
python scripts/train.py \
  --config configs/paper_manuscript_controlled_2048.json \
  --train-root data/prepared/remove_75/train \
  --output runs/paper/remove_75 \
  --device cuda:0
```

### Training output

A typical run directory contains:

```text
runs/paper/remove_50/
├── resolved_config.json
├── run_metadata.json
├── history.jsonl
├── training_result.json
├── checkpoints/
│   ├── latest.pt
│   ├── best.pt
│   ├── epoch_0100.pt
│   ├── epoch_0200.pt
│   ├── epoch_0300.pt
│   └── epoch_0400.pt
└── samples/
```

Checkpoint files are generated according to the configuration's save interval.

> [!IMPORTANT]
> When no validation loader is supplied, the training code selects `best.pt` using training loss. In that case, `best.pt` is **not validation-best**. For a no-validation run, use a predeclared checkpoint-selection rule such as the final saved epoch, and report that rule explicitly. Do not select a checkpoint using test-set performance.

### GPU memory and configuration changes

The manuscript configuration uses batch size 8. Actual memory use depends on the GPU, PyTorch build, and runtime environment. The training CLI accepts batch-size and worker overrides, but changing them creates a modified training run. Record any override in the experiment metadata and do not silently describe the run as an exact configuration match.

---

## Evaluation

The maintained evaluation script uses the full ancestral reverse-diffusion process.

### Evaluate all three controlled settings

After training, evaluate the three severity-specific models and aggregate their results:

```bash
python scripts/evaluate_controlled_suite.py \
  --prepared-root data/prepared \
  --runs-root runs/paper \
  --output evaluations/paper \
  --severities 25 50 75 \
  --checkpoint-name epoch_0400.pt \
  --device cuda:0 \
  --weights raw \
  --seed 42 \
  --repeats 1 \
  --f-threshold 0.02 \
  --cd-scale 10000
```

The command uses the final 400th-epoch checkpoint. If you trained with a validation split and intentionally select the validation-best checkpoint, use `--checkpoint-name best.pt` and document the validation-selection rule.

The suite writes per-severity summaries and aggregate files, including:

```text
evaluations/paper/
├── remove_25/
│   ├── per_sample.csv
│   └── summary.json
├── remove_50/
│   ├── per_sample.csv
│   └── summary.json
├── remove_75/
│   ├── per_sample.csv
│   └── summary.json
├── controlled_summary.csv
└── controlled_summary.json
```

### Evaluate one checkpoint manually

```bash
python scripts/evaluate.py \
  --checkpoint runs/paper/remove_50/checkpoints/epoch_0400.pt \
  --data-root data/prepared/remove_50/test \
  --output evaluations/paper/remove_50 \
  --device cuda:0 \
  --batch-size 1 \
  --num-workers 2 \
  --weights raw \
  --seed 42 \
  --repeats 1 \
  --f-threshold 0.02 \
  --cd-scale 10000
```

### Evaluation metrics

- **CD-L1:** symmetric unsquared Euclidean Chamfer distance.
- **CD-L2:** symmetric squared-Euclidean Chamfer distance.
- **Precision and recall:** nearest-neighbour distance threshold based.
- **F-score:** harmonic mean of precision and recall at the selected threshold.

For the controlled manuscript comparison, use an F-score threshold of 0.02 and report CD-L2 with the 10⁴ display scale. The scale factor changes presentation only; raw CD-L2 is also saved.

The training reconstruction loss and evaluation CD-L2 are not the same metric definition.

### Evaluation provenance

For every evaluation, record:

- checkpoint path and hash;
- raw or EMA weight selection;
- dataset split and manifest;
- evaluation seed and repeat count;
- F-score threshold;
- CD-L2 display scale;
- resolved model configuration;
- PyTorch, CUDA, and GPU versions.

See [`docs/EVALUATION.md`](docs/EVALUATION.md) for additional details.

> **Benchmark note:** The public release contains the controlled-data preparation and evaluation pipeline. Official Completion3D and ShapeNet-55 data are not redistributed here. The benchmark tables in this README are transcribed from the manuscript and are not claimed to have been regenerated from this ZIP.

---

## Inference

Inference requires a partial point cloud and a trained checkpoint. The script resamples the partial observation to the checkpoint's expected input size and derives the FPS structure view internally.

```bash
python scripts/infer.py \
  --checkpoint runs/paper/remove_50/checkpoints/epoch_0400.pt \
  --partial path/to/partial_cloud.txt \
  --output outputs/completed_cloud.txt \
  --device cuda:0 \
  --weights raw \
  --seed 42
```

Optional normalization:

```bash
--normalize-partial
```

Use this flag only when normalization matches the preprocessing used to train the checkpoint.

The inference script saves:

- the completed point cloud;
- the prepared partial point cloud;
- the FPS structure point cloud;
- a JSON metadata file describing the checkpoint formulation and selected weights.

No complete target is needed at inference.

---

## Diagnostics and tests

### Fixed-timestep diagnostics

The diagnostic script measures denoising behavior at selected diffusion timesteps and reports reconstruction metrics and token-level routing statistics.

```bash
python scripts/diagnose_timesteps.py \
  --checkpoint runs/paper/remove_50/checkpoints/epoch_0400.pt \
  --data-root data/prepared/remove_50/test \
  --output diagnostics/remove_50 \
  --weights raw \
  --timesteps 0 50 100 200 300 350 375 399 \
  --cd-scale 10000
```

For a small diagnostic run, use the optional `--limit` argument. Do not use a limited run as a full benchmark evaluation.

### Dataset inspection

```bash
python scripts/dataset_report.py \
  --manifest data/prepared/manifest.csv \
  --output diagnostics/dataset_report.json \
  --require-main-method
```

### Run the automated test suite

Install the development dependencies first, then run:

```bash
python -m pytest -q
```

### Compile-check the source

```bash
python -m compileall -q src scripts tests
```

### Check CLI arguments

The scripts expose their supported options through `--help`:

```bash
python scripts/make_split_manifest.py --help
python scripts/prepare_dataset.py --help
python scripts/validate_dataset.py --help
python scripts/train.py --help
python scripts/train_controlled_suite.py --help
python scripts/evaluate.py --help
python scripts/evaluate_controlled_suite.py --help
python scripts/infer.py --help
python scripts/diagnose_timesteps.py --help
```

---

## Manuscript configuration and experimental branches

The repository preserves multiple diffusion configurations for research traceability. They must not be treated as interchangeable.

| Configuration | Prediction target | Forward schedule | Reverse variance | Training epochs | Role |
|---|---|---|---|---:|---|
| `paper_manuscript_controlled_2048.json` | Epsilon | Linear | Fixed beta | 400 | **Manuscript-aligned controlled configuration** |
| `paper_controlled_2048.json` | Epsilon | Linear | Fixed beta | Historical | Earlier paper-control configuration |
| `legacy_v020_2048.json` | Epsilon | Linear | Fixed beta | Historical | Legacy baseline |
| `v030_recommended_2048.json` | V-prediction | Zero-terminal-SNR rescaled linear | Posterior | Experimental | Earlier v0.3 proposal |
| `v032_recommended_2048.json` | V-prediction | Experimental | Posterior | Experimental | Numerical-stability branch |
| `v033_recommended_2048.json` | V-prediction | Experimental | Posterior | Experimental | CUDA/KNN development branch |
| `v034_recommended_2048.json` | V-prediction | Experimental | Posterior | Experimental | Stable-geometry development branch |
| `v035_stable_vpred_2048.json` | V-prediction | Linear | Posterior | 1,500 | Later stability-first experiment |
| `debug_smoke.json` | Debug setting | Short debug schedule | Debug setting | 1 | Small smoke-test configuration |

For the method and training setup described in the manuscript, use `paper_manuscript_controlled_2048.json`.

The v0.3.5 configuration is a later experimental branch. Results from it must be labeled separately and must not be attributed to the manuscript configuration unless the manuscript is updated to match that exact formulation and its measured results.

See [`docs/SCIENTIFIC_ALIGNMENT.md`](docs/SCIENTIFIC_ALIGNMENT.md), [`docs/PAPER_REPRODUCTION.md`](docs/PAPER_REPRODUCTION.md), and [`CHANGELOG.md`](CHANGELOG.md).

---

## Oracle structural-condition experiment

The deployable method derives the FPS structure view from the partial observation.

The repository also supports a privileged structural-condition ablation derived from the complete target. This uses information that is unavailable at inference and must be labeled **oracle**, **privileged**, and **non-deployable**.

The preparation script requires explicit acknowledgement:

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

Never report this oracle setting as the deployable TARGDiff inference condition.

---

## Reproducibility

For every experiment intended for comparison or publication, preserve the following together:

- repository commit hash;
- exact source-object split manifest;
- split-manifest SHA-256;
- raw dataset release/version information;
- prepared-dataset metadata;
- resolved configuration;
- runtime metadata;
- training command and log;
- checkpoint and checkpoint hash;
- checkpoint-selection rule;
- evaluation command;
- evaluation seed and repeat count;
- raw and scaled metric values;
- prediction files when required.

The preparation pipeline records a manifest fingerprint and dataset metadata. The fingerprint identifies the manifest bytes; it does not replace the need to record the original raw dataset version.

### Release and scientific documentation

- [Paper reproduction guide](docs/PAPER_REPRODUCTION.md)
- [Data preparation guide](docs/DATA_PREPARATION.md)
- [Evaluation guide](docs/EVALUATION.md)
- [Scientific alignment across configurations](docs/SCIENTIFIC_ALIGNMENT.md)
- [Code audit](docs/CODE_AUDIT.md)
- [Reproducibility checklist](docs/REPRODUCIBILITY.md)
- [Public release audit](docs/PUBLIC_RELEASE_AUDIT.md)
- [Release verification record](docs/GITHUB_RELEASE_VERIFICATION.md)
- [Publication release checklist](PUBLICATION_RELEASE_CHECKLIST.md)
- [GitHub setup guide](GITHUB_SETUP.md)

### Important result-provenance statement

The numerical results in this README are copied from the manuscript. They should not be described as independently reproduced by the public source release unless the exact paper split, code commit, configuration, checkpoint, and evaluation outputs have been recovered and verified together.

---

## Limitations

The current manuscript and implementation identify the following limitations:

- The 64 local condition tokens are selected by point-array order rather than a permutation-invariant geometric selector.
- Structural-loss subsets also depend on point-array order.
- The controlled directional crops do not reproduce all real-sensor effects, including sensor noise, non-uniform density, and complex occlusion.
- The spatial and temporal evolution of learned routing weights is not fully analyzed in the manuscript.
- The controlled 25%, 50%, and 75% settings use separately trained models; they do not demonstrate cross-severity generalization by a single model.
- Completion3D, ShapeNet-55, and the controlled PCN-derived experiment use different protocols and should not be treated as one interchangeable evaluation.
- The reduced-step DDIM latency table is reported in the manuscript, but the maintained public evaluation CLI in this release uses the full ancestral reverse chain.
- Exact paper checkpoints and the audited source split manifest are not included in this source ZIP.

These limitations should remain explicit when interpreting the results or extending the method.

---

## License

No open-source license has been selected in this repository snapshot. The repository includes [`LICENSE_NOT_SELECTED.md`](LICENSE_NOT_SELECTED.md) to make this status explicit.

Until a license is selected and included, public visibility of the repository does not grant permission to reuse, modify, or redistribute the code.

Before choosing a license, the project owners should confirm institutional rights, third-party code obligations, dataset restrictions, and the provenance of retained legacy files.

---

## Citation

If you use TARGDiff in academic work, please cite the associated paper when its author list and publication information are finalized.

The repository includes `CITATION.cff.template`. Complete the author and publication metadata, then rename it to `CITATION.cff` before the public release.

BibTeX template:

```bibtex
@article{targdiff2026,
  title   = {TARGDiff: Time-Adaptive Routing-Gated Diffusion for 3D Point Cloud Completion},
  author  = {<FINAL AUTHOR LIST>},
  journal = {<JOURNAL, CONFERENCE, OR PREPRINT>},
  year    = {2026},
  url     = {<PAPER OR PROJECT URL>}
}
```

---

<p align="center">
  <strong>TARGDiff — Time-Adaptive Routing-Gated Diffusion for 3D Point Cloud Completion</strong><br>
  <sub>Research code, experimental records, and reproducibility documentation.</sub>
</p>
