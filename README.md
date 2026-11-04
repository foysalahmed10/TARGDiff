<h1 align="center">TARGDiff</h1>

<p align="center">
  <strong>Time-Adaptive Routing-Gated Diffusion for 3D Point Cloud Completion</strong>
</p>

<p align="center">
  A diffusion-based framework for completing partial 3D point clouds through time-adaptive fusion of complementary views of the observed geometry.
</p>

<p align="center">
  <img src="assets/fig1_overview.png" alt="Overview of the TARGDiff framework" width="900">
</p>

## Overview

TARGDiff addresses 3D point cloud completion by adapting how complementary geometric observations are used throughout the reverse-diffusion process.

The method builds two representations from the **same partial input**: the dense observation and a compact, 256-point farthest-point-sampled (FPS) view. The dense view preserves local sampling detail, while the FPS view provides broader coverage of the visible shape.

Its central component, **TARGFuse**, independently aligns both views with the current denoising query and predicts token-wise routing weights using the query and diffusion timestep. Explicit disagreement between the two contexts is used after routing for residual refinement.

The complete point cloud is used for training supervision and evaluation only. It is not an inference-time condition.

## Key Contributions

- **Complementary observation views:** Dense and FPS-based representations are derived from the same partial point cloud.
- **Time-adaptive routing:** TARGFuse adjusts the relative contribution of the two views for each output token during denoising.
- **Disagreement-aware refinement:** Explicit differences between the two contextual representations are used after routing.
- **Broad evaluation:** The study includes controlled missingness experiments, Completion3D, ShapeNet-55, component ablations, and reduced-step sampling.

## Results

### Controlled missingness

TARGDiff is evaluated under matched, severity-specific training at three levels of missing geometry.

| Missing geometry | CD-L2 (×10⁴) ↓ | F-score@0.02 ↑ |
|:---:|---:|---:|
| 25% | **8.71** | **0.896** |
| 50% | **12.43** | **0.760** |
| 75% | **17.34** | **0.573** |

Compared with the strongest listed CD baseline, the relative CD-L2 reductions are 3.5%, 5.3%, and 12.0%, respectively.

**Protocol note:** A separate model is trained for each missingness level. These results therefore do not represent cross-severity generalization by a single model.

### Standard benchmarks

| Benchmark | Result |
|---|---|
| Completion3D | Average CD: **7.14** (×10⁴) |
| ShapeNet-55 | CD-L2 (×10³): Simple **0.43**, Moderate **0.62**, Hard **1.08**, Average **0.71** |
| ShapeNet-55 | F-score@1%: **0.525** |

On ShapeNet-55, TARGDiff achieves the lowest listed CD-L2 under the Hard setting. It does not lead every aggregate metric.

## Qualitative Results

The following figures show comparisons at different levels of missing geometry:

- [25% missing geometry](assets/fig4_qualitative_25.png)
- [50% missing geometry](assets/fig5_qualitative_50.png)
- [75% missing geometry](assets/fig6_qualitative_75.png)

Additional figures: [Detailed architecture](assets/fig2_architecture.png) · [Controlled missingness protocol](assets/fig3_protocol.png)

## Code and Reproducibility

The paper-matched source code and experiment configurations are being prepared for release in this repository. Installation, dataset preparation, training, and evaluation instructions will be provided with the corresponding code release.

## Citation

Citation information will be added when the author list and publication details are finalized.

---

<p align="center">
  <sub>TARGDiff — 3D Point Cloud Completion</sub>
</p>
