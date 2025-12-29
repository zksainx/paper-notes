# TritonForge: Profiling-Guided Framework for Automated Triton Kernel Optimization

<div class="paper-meta" markdown>

**Authors**: Haonan Li, Keyu Man, Partha Kanuparthy, Hanning Chen, Wei Sun, Sreen Tallam, Chenguang Zhu, Kevin Zhu, Zhiyun Qian  
**Institution**: UC Riverside, Meta, UC Irvine  
**Conference**: arXiv 2025  
**Paper Link**: [arXiv:2512.09196](https://arxiv.org/abs/2512.09196)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Triton</span>
<span class="paper-tag">LLM for Code</span>
<span class="paper-tag">GPU Kernel Optimization</span>
<span class="paper-tag">Profiling</span>
</div>

## Abstract

TritonForge is a profiling-guided framework for automated Triton kernel optimization that integrates kernel analysis, runtime profiling, and iterative code transformation. By incorporating feedback from Nsight Compute profiling results, the system identifies performance bottlenecks, proposes targeted code modifications, and evaluates their impact automatically.

**Key Contributions**:

- LLM-guided kernel generation and optimization with structured profiling feedback
- Profiling-aware code improvements translating low-level metrics into actionable transformations
- Automated performance evaluation pipeline reducing reliance on manual profiling expertise
- Achieves up to 5x performance improvement over baseline, with 42.7% success rate on TritonBench

## Core Ideas

### Architecture Overview

TritonForge consists of two main stages:

1. **Test Generator**: LLM agent that understands Triton code semantics and generates performance tests under specific NVTX range labels
2. **Kernel Optimizer**: LLM agent that reads performance metrics from Nsight Compute and generates optimized Triton code

Additional component: **Fault-Aware Remediation Agent** for reviewing and fixing compilation/runtime errors.

### Optimization Pipeline

The kernel optimization loop includes:

| Component | Input | Output | Role |
|-----------|-------|--------|------|
| Initialization-and-Proposal | Device capabilities, kernel source, performance report | Proposal kernel | Generate candidate refinements (tiling, num_warps, vectorization, memory layout) |
| Fault-Aware Remediation | Kernel, diagnostic logs | Repaired kernel | Fix compilation/runtime failures using LLM edits |
| Performance-Arbiter | Profiler report, best-so-far report | Decision (accept/continue/finish) | Multi-criteria decision based on speedup, robustness, resource limits |
| Targeted-Refinement-Hint | Previous and new reports | Refined kernel | Convert profiler deltas into actionable feedback |

### Profiling Integration

Uses NVIDIA Nsight Compute to collect hardware-level metrics:
- Memory throughput and efficiency
- Compute throughput
- Kernel occupancy
- L2 cache throughput

NVTX markers isolate the Triton kernel region of interest for precise profiling.

### Case Study: GEMV Restructuring

**Baseline bottleneck** (identified via profiling):

- Memory throughput: 52.24% of peak
- Compute throughput: 5.92% (ALUs under-utilized)
- Occupancy: 37.5%

**TritonForge optimization**: Restructured from broadcast-based expansion to GEMV-style streamed reduction:
```python
# Before: 3D broadcast prior to reduction
expanded_a, _ = tl.broadcast(a, b)
acc += tl.trans(tl.sum(expanded_a * b, axis=2))

# After: Tile-wise accumulation
acc[n0:n0+Nt] += tl.sum(b_tile * a_tile[None, :], axis=1)
```

**Results after optimization**:

- Memory throughput: 90.76%
- Compute throughput: 15.00%
- L2 Cache throughput: 71.52%
- **1.74x speedup**

## Experimental Results

**Platform**: NVIDIA H100 GPU (96GB), Triton 3.4, Gemini-2.5-Pro

**Overall Results on TritonBench (131 kernels)**:

| Category | Success Rate | Avg. Speedup |
|----------|-------------|--------------|
| Q1 (shortest) | 42.3% | 2.25x |
| Q2 | 40.0% | 1.40x |
| Q3 | 42.8% | 1.62x |
| Q4 (longest) | 45.5% | 2.05x |
| **Overall** | **42.7%** | **1.76x** |

**Ablation Study** (36 kernels):

| Method | Avg. Speedup | Success Rate |
|--------|-------------|--------------|
| TritonForge (full) | 1.51x | 42.3% |
| LLM E2E (w/o profiling) | 1.33x | 22.8% |
| LLM One-shot | 2.91x | 11.4% |

## Limitations

- Iterative optimization may get stuck in semantically-equivalent but performance-neutral variants
- LLM exploration space is shallow once obvious structural fixes are exhausted
- Absence of strong gradient-like feedback from profiling signals limits convergence

---

*Reading date: 2025-12*
*Note status: Completed*
