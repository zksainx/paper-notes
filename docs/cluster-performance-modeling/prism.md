# PRISM: Probabilistic Runtime Insights and Scalable Performance Modeling for Large-Scale Distributed Training

<div class="paper-meta" markdown>

**Authors**: Alicia Golden, Michael Kuchnik, Samuel Hsia, Zachary DeVito, Gu-Yeon Wei, David Brooks, Carole-Jean Wu  
**Institution**: FAIR at Meta, Harvard University  
**Conference**: arXiv  
**Year**: 2025  
**Paper Link**: [arXiv:2510.15596](https://arxiv.org/abs/2510.15596)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Distributed Training</span>
<span class="paper-tag">Performance Modeling</span>
<span class="paper-tag">Tail Latency</span>
</div>

## Background

Large-scale training has entered a regime where runtime noise is structural rather than exceptional. When jobs scale to tens of thousands of GPUs, per-kernel variance, communication stragglers, thermal throttling, and topology contention compose into step-time tails that dominate real wall-clock cost. In a 64K+ GPU production run, the paper reports around 9% step-time variability and estimates this can accumulate into roughly 20 extra days for a frontier training job.

The core gap is that many existing models optimize for mean runtime as if execution were deterministic. That assumption is increasingly fragile for modern clusters where synchronization barriers amplify outliers: a single slow rank can stall everyone else. PRISM reframes prediction as a distributional problem and directly targets p95 guarantees for system planning.

**Key Takeaways**

- Runtime variability in production training clusters is measurable, large, and scale-amplified rather than negligible noise.
- PRISM models end-to-end step time as a stochastic DAG composition problem, not a single-point estimate.
- p95-aware predictions are useful beyond model tuning, including placement decisions and cluster scheduling.

## Methodology

PRISM combines trace structure with statistical duration modeling. The framework ingests workload graph structure, parallelization configuration, and cluster topology, then runs Monte Carlo simulation over a multi-rank DAG to estimate step-time distributions.

A key design choice is category-specific modeling of variability. Compute kernels, collectives, and pipeline point-to-point communication each have different variance behavior and therefore use different pooling and sampling rules. This avoids collapsing fundamentally different mechanisms into one generic noise model.

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Context Parser | Model graph, DP/TP/PP/CP config, topology metadata | Structured runtime context | Defines where operations execute and which network paths they use |
| DAG Builder | One profiled multi-rank trace | Dependency graph of one training step | Captures serial edges and synchronization dependencies |
| Duration Pool Builder | Repeated traces or profiled operator samples | Per-signature duration distributions | Builds empirical/statistical runtime pools |
| Distribution Fitter | Per-signature samples | Fitted Normal/Gamma/LogNormal candidates | Uses KS-based fitting for kernel-duration modeling |
| Variability-Aware Sampler | DAG nodes + duration pools + communication regime | One sampled step instance | Draws correlated durations under shared network conditions |
| Monte Carlo Engine | Repeated sampled DAG instances (N=1000) | Step-time distribution and p95 estimate | Produces probabilistic runtime guarantees |

### Operation-Specific Modeling Rules

| Operation Type | Signature / Grouping | Sampling Policy | Why It Matters |
| --- | --- | --- | --- |
| Compute kernels | Kernel name + tensor shape + stream | Sampled from per-kernel pool | Captures temporal/spatial kernel variance |
| Collectives (AllReduce/AllGather/ReduceScatter) | Includes process group (TP/DP/CP) and topology context | One sampled duration broadcast to all participants | Respects collective blocking semantics and topology effects |
| Pipeline Send/Recv | Position-aware classification into ever-bubble vs never-bubble | Otsu-threshold split + regime-conditioned sampling | Separates transfer latency from pipeline-bubble waiting |

### Key Design Choices

- Parallelization-aware and topology-aware signatures prevent mixing durations from different communication domains.
- Communication events in a step are sampled under a shared regime to preserve correlation under congestion-like conditions.
- DAG composition uses sum on serial dependencies and max on parallel branches, matching distributed synchronization behavior.
- Bubble-sensitive Send/Recv handling avoids unrealistic sampling for pipeline-heavy schedules.
- The same framework supports measured pools and parametric extrapolation for design-space exploration.

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Primary objective | Validate p95 step-time prediction accuracy under real variability |
| Dataset scale | 1,092 real training runs |
| Split | 344 train / 748 test runs |
| GPU-scale settings | 64-GPU sweeps + one 64K+ GPU production trace |
| Parallelism coverage | DP/TP/PP configurations + multiple pipeline schedules |
| Simulation runs | Monte Carlo N=1000 per scenario |
| Additional system study | 1,280-GPU scheduler simulation with Philly traces |

### Headline Results

| Setting | Result |
| --- | --- |
| Overall PRISM validation (748-run test set) | p95 prediction error within 4.3% |
| Example config A validation | p95 error 0.818% |
| Observed 64K+ production step-time distribution | p5/p50/p95 = 18.6s / 18.8s / 19.3s |
| Spatial GEMM variation across 20K+ GPUs | 1.64%-14.04% |
| Temporal GEMM variation (single GPU repeated runs) | 0.98%-6.46% |
| Inter-node collective tails | Up to order-of-magnitude higher than mean |

The accuracy numbers show PRISM is not only descriptive but predictive for tail behavior. The measured production spread at 64K+ GPUs also supports the central claim that average-case modeling leaves significant risk unmodeled.

### System-Design Insights Enabled by PRISM

| Analysis Scenario | Finding | Practical Implication |
| --- | --- | --- |
| Localized hot-node slowdown | Single degraded node can raise step time by 9%-35% | Node health anomalies materially affect global throughput |
| Slow-rank placement across PP stages | Same slowdown magnitude yields 8.0%-15.6% p95 difference by placement | Rank/stage mapping should be variability-aware |
| Kernel-level variance attribution | High-variance kernels are not always high-impact | Critical-path position matters as much as raw variance |
| Scheduler-level p95 estimates | JCT reduced to 0.3x-0.8x vs point-estimate baseline for estimate-using schedulers | p95-aware planning improves cluster efficiency under uncertainty |

The ablation results are particularly important: PRISM shows variance contribution is not additive and cannot be inferred from kernel CV ranking alone. Dependencies and max-path competition in the DAG decide which variability actually hits end-to-end p95.

## Limitation

The framework is strong on realism but still carries evidence and deployment constraints.

| Limitation | Why It Matters |
| --- | --- |
| High validation cost | 1,092-run evidence collection is expensive, which can limit frequent re-calibration |
| Data dependency on trace quality | Inaccurate or incomplete traces can bias duration pools and tail estimates |
| Some assumptions in scheduling study | CV settings and simulator abstractions may not match every production cluster |
| Limited public reproducibility at publication time | Code release is described as post-acceptance, reducing immediate external verification |
| Tail guarantees are percentile-specific | Strong p95 prediction does not automatically guarantee p99/p999 behavior |

---

*Reading date: 2026-04*
*Note status: Completed*
