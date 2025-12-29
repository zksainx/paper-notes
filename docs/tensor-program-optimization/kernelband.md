# KernelBand: Boosting LLM-based Kernel Optimization with a Hierarchical and Hardware-aware Multi-armed Bandit

<div class="paper-meta" markdown>

**Authors**: Dezhi Ran, Shuxiao Xie, Mingfang Ji, Ziyue Hua, Mengzhou Wu, Yuan Cao, Yuzhe Guo, Yu Hao, Linyi Li, Yitao Hu, Tao Xie  
**Institution**: Peking University, ECNU, Tianjin University, Beijing Jiaotong University, HKUST, Simon Fraser University  
**Conference**: arXiv 2025  
**Paper Link**: [arXiv:2511.18868](https://arxiv.org/abs/2511.18868)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">LLM for Code</span>
<span class="paper-tag">Multi-armed Bandit</span>
<span class="paper-tag">Kernel Optimization</span>
<span class="paper-tag">Triton</span>
</div>

## Abstract

KernelBand formulates kernel optimization as a hierarchical multi-armed bandit (MAB) problem, enabling LLM agents to strategically navigate the optimization space by treating kernel selection and optimization strategy application as sequential decision-making processes. The approach leverages hardware profiling to identify promising strategies and employs runtime behavior clustering to reduce exploration overhead.

**Key Contributions**:
- First formulation of kernel optimization as hierarchical MAB problem
- Profiling-guided strategy selection with runtime behavior clustering
- Superior performance on TritonBench with fewer tokens and consistent improvement without saturation

## Core Ideas

### Problem Formulation

**Kernel Optimization Problem**: Given operation O and hardware H, find implementation k* that minimizes execution time:

$$
k^* = \arg\min_{k \in \mathcal{K}} \mathcal{T}(k, \mathcal{H})
$$

**Hierarchical MAB Formulation**:

- At iteration t, agent selects: (i) kernel $k_t \in C_t$, (ii) strategy $s_t \in S$
- Action: $a_t = (k_t, s_t)$ treated as single arm
- Reward: $r_t = 1 - \frac{\mathcal{T}(k_{t+1}, \mathcal{H})}{\mathcal{T}(k_t, \mathcal{H})}$ (relative speedup)
- Failed kernels receive $r_t = -1$

### Three Key Mechanisms

**1. Profiling-Guided Strategy Selection**

9-dimensional profiling vector per kernel:
```
φ(k) = [L2_hit, mem_bw, sm_util, warp_eff, achieved_occupancy,
        reg_per_thread, shared_conflicts, load_store_coalesced,
        tensor_core_util]
```

Each strategy s maintains logistic head for compatibility:

$$
\psi(s, \phi(k)) = \sigma(w_s^T \phi(k) + b_s) \in [0,1]
$$

**2. Runtime Behavior Clustering**

6-dimensional runtime feature vector:
```
ρ(k) = [log T(k), log mem_footprint(k), arithmetic_intensity(k),
        block_dimx, block_dimy, achieved_occupancy]
```

Partition into K=3 clusters using incremental k-means++, enabling knowledge transfer across similar kernels.

**3. Hierarchical UCB Algorithm**

Three-term UCB surrogate:

$$
\text{UCB}_{c(k),s}(t) = \underbrace{\hat{\mu}_{c(k),s}}_{\text{exploitation}} + \underbrace{\sqrt{\frac{2\ln t}{n_{c(k),s}}}}_{\text{exploration}} + \underbrace{\alpha \psi(s, \phi(k))}_{\text{profiling bias}}
$$

### Algorithm Overview

```
1: Initialize candidate pool C₁ = {k₀}
2: for t = 1 to T do
3:   Cluster Cₜ into K behavior-based clusters
4:   Extract profiling features φ(k) for all k ∈ Cₜ
5:   Select (kₜ, sₜ) = argmax UCB score
6:   kₜ₊₁ ← LLM(kₜ, sₜ)  // Apply strategy via LLM
7:   rₜ ← Evaluate(kₜ₊₁, H)
8:   Update cluster-level statistics
9:   if kₜ₊₁ valid then Cₜ₊₁ ← Cₜ ∪ {kₜ₊₁}
10: end for
11: return argmin T(k, H) over k ∈ C_T
```

### Regret Guarantee

Under cluster-similarity assumption:

$$
\mathbb{E}[\text{Regret}(T)] \leq 3|S|\sqrt{8T \ln T} + T\epsilon_{\text{cluster}} + T\alpha\epsilon_{\text{profile}}
$$

Scales with number of clusters (3) rather than pool size |Cₜ|.

## Experimental Results

**Setup**: RTX 4090 / A100, DeepSeek-V3.2 / GPT-5.1, TritonBench L1+L2 (29 kernels)

**Main Results** (Best-of-iteration, RTX 4090 + DeepSeek-V3.2):

| Method | Iter 0 | Iter 3 | Iter 6 | Iter 9 |
|--------|--------|--------|--------|--------|
| GEAK | 0.70x/7% | 1.66x/25% | 2.15x/25% | 2.40x/32% |
| **KernelBand** | **2.28x/29%** | **3.15x/46%** | **7.54x/46%** | **9.78x/54%** |

*(Format: Speedup/Fast@1)*

**Cross-Platform Results** (Iter 9, Best-of-iteration):

| Platform | GEAK | KernelBand |
|----------|------|------------|
| 4090 + DeepSeek | 2.40x/32% | 9.78x/54% |
| 4090 + GPT-5.1 | 1.13x/24% | 12.06x/52% |
| A100 + DeepSeek | 3.22x/21% | 22.30x/72% |

**Scalability** (20 iterations): KernelBand continues improving while GEAK saturates around iteration 10.

### Case Study: seeded_dropout Kernel

**Optimization progression**:

1. Baseline: fixed BLOCK_SIZE configuration
2. First breakthrough: autotuning with adaptive block sizing and warp-level parallelism
3. Second breakthrough: vectorized loads/stores with tunable vector widths
4. Refinements: simplified vectorized memory access to reduce instruction overhead

**Key advantages demonstrated**:

- Hierarchical MAB prevented catastrophic regression by maintaining diverse candidates
- Profiling-guided selection identified memory bandwidth as primary bottleneck
- Runtime clustering enabled knowledge transfer from similar memory-bound kernels

## Limitations & Future Work

- Distinctive challenges: non-stationary rewards, contextual strategy effectiveness, large action space, expensive evaluations
- Future: adaptive stopping criteria, diversity-driven re-prompting, memory-augmented iterative reasoning

---

*Reading date: 2025-12*
*Note status: Completed*
