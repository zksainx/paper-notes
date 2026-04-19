# TritonForge: Profiling-Guided Framework for Automated Triton Kernel Optimization

<div class="paper-meta" markdown>

**Authors**: Haonan Li, Keyu Man, Partha Kanuparthy, Hanning Chen, Wei Sun, Sreen Tallam, Chenguang Zhu, Kevin Zhu, Zhiyun Qian  
**Institution**: UC Riverside, Meta, UC Irvine  
**Conference**: arXiv  
**Year**: 2025  
**Paper Link**: [arXiv:2512.09196](https://arxiv.org/abs/2512.09196)  
**GitHub**: [RLsys-Foundation/TritonForge](https://github.com/RLsys-Foundation/TritonForge)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Triton</span>
<span class="paper-tag">Automated Kernel Generation</span>
<span class="paper-tag">GPU Kernel Optimization</span>
</div>

## Background

TritonForge targets a practical bottleneck in Triton kernel development. Triton makes GPU programming more approachable than raw CUDA, but high performance still depends on hardware-aware choices around tiling, `num_warps`, `num_stages`, memory layout, occupancy, and register pressure. Functional kernels are relatively easy to write; fast kernels still require specialized optimization knowledge.

The main gap is the missing loop between runtime evidence and code revision. Manual Triton optimization usually alternates between profiling, bottleneck diagnosis, parameter tuning, and repeated evaluation. Static LLM-based code optimization can generate plausible kernels, but it often misses the dominant hardware bottleneck because source code alone does not reveal the real execution constraints. TritonForge closes this loop by feeding profiling signals back into iterative kernel rewriting.

**Key Takeaways**

- TritonForge treats kernel optimization as an agentic feedback loop rather than a one-shot code generation problem.
- Runtime profiling is the main signal that separates high-coverage optimization from static code-only prompting.
- The framework can occasionally move beyond parameter tuning into structural rewrites, such as converting broadcast-heavy code into streamed GEMV-style reduction.

## Methodology

TritonForge is organized as a profiling-guided optimization workflow around Triton kernels. The system combines test generation, runtime profiling, code rewriting, and error remediation so that each optimization round is informed by concrete hardware-level evidence rather than only source-level heuristics.

The benchmark substrate is TritonBench, which provides 184 real-world Triton kernels collected from GitHub repositories. TritonForge extends it with correctness and performance tests, hardware specifications, and profiling traces. That augmentation gives the model access to code semantics, execution behavior, and device context at the same time.

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Test Generator | Input Triton kernel and existing tests | Performance-oriented test cases with NVTX labels | Creates executable workloads that can be profiled consistently |
| Profiling Module | Generated tests and candidate kernel | Runtime latency plus Nsight Compute metrics | Collects memory, compute, and occupancy signals |
| Kernel Optimizer | Kernel source, hardware context, profiling report | Revised Triton kernel | Proposes targeted code changes based on measured bottlenecks |
| Fault-Aware Remediation | Failed candidate and diagnostic logs | Repaired kernel | Fixes compilation or runtime failures introduced by previous rewrites |
| Performance Arbiter | New profiler report and best-so-far result | Accept / continue / finish decision | Stops low-value iterations and keeps the best candidate |

### Key Design Choices

- Profiling is not an afterthought; it is the central control signal used to decide how the kernel should change.
- The optimization loop is iterative and bounded. The paper caps optimization at 8 iterations to avoid non-terminating repair cycles.
- The feedback vocabulary is hardware-facing: memory throughput, compute throughput, occupancy, instruction stalls, and L2 behavior are translated into concrete edits such as retuning tiling, reducing register pressure, or changing memory access patterns.
- The implementation uses Gemini-2.5-Pro as the reasoning model, about 3,000 lines of Python for orchestration, and roughly 300 lines of profiling automation.

### Optimization Pipeline

| Stage | Main Inputs | Main Outputs | Typical Action |
| --- | --- | --- | --- |
| Initialization-and-Proposal | Device capabilities, kernel source, performance report | Candidate kernel | Adjust tiling, block shapes, `num_warps`, vectorization, memory layout, or loop structure |
| Fault-Aware Remediation | Candidate kernel and error logs | Repaired kernel | Repair compile-time or runtime failures with targeted edits |
| Performance-Arbiter | Current report, best-so-far report, thresholds | Continue / accept / finish | Compare throughput, latency, robustness, and diminishing returns |
| Targeted-Refinement-Hint | Previous and current reports | Actionable refinement hint | Convert profiler deltas into next-step optimization guidance |

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Hardware | NVIDIA H100 GPU (96GB) |
| Model | Gemini-2.5-Pro |
| Profiler | NVIDIA Nsight Compute 2025.2.1.0 |
| Benchmark | TritonBench |
| Full Benchmark Size | 184 real-world Triton kernels |
| Runnable Kernels | 131 |
| Success Criterion | At least 1.05x speedup over baseline |
| Average Optimization Rounds | 3.2 |
| Average Wall-Clock Time | 22 minutes per kernel |
| Average API Cost | $1.20 per kernel |

### Headline Results

| Category | # Kernels | Success Rate | Avg. Speedup |
| --- | --- | --- | --- |
| Q1 | 26 | 42.3% | 2.25x |
| Q2 | 35 | 40.0% | 1.40x |
| Q3 | 35 | 42.8% | 1.62x |
| Q4 | 22 | 45.5% | 2.05x |
| Tail High | 4 | 50.0% | 1.16x |
| Tail Low | 9 | 66.7% | 1.79x |
| **Overall** | **131** | **42.7%** | **1.76x** |

The full benchmark results show moderate but meaningful coverage. Success rates remain fairly stable across kernel-length buckets, while the absolute gain varies more. Q1 delivers the highest average speedup, which suggests that shorter kernels expose more low-hanging optimization opportunities.

### Ablation Study

| Method | Avg. Speedup on Success | Success Rate | Avg. Iterations |
| --- | --- | --- | --- |
| TritonForge | 1.51x | 42.3% | 3.8 |
| LLM E2E (without profiling) | 1.33x | 22.8% | 3.0 |
| LLM One-shot Prompt | 2.91x | 11.4% | 1.0 |

The ablation results make the value of profiling very clear. One-shot prompting occasionally finds a large win, but only on a very small subset of kernels. Removing execution feedback cuts the success rate roughly in half. The full system gives up some peak one-shot upside in exchange for far better reliability across a heterogeneous benchmark.

### Profiling-Guided Case Study

| Performance Factor | Before | After |
| --- | --- | --- |
| Memory Throughput | 52.24% | 90.76% |
| Compute Throughput | 5.92% | 15.00% |
| L2 Cache Throughput | 50.04% | 71.52% |
| Kernel Runtime | 849.50 us | 488.96 us |
| Speedup | 1.00x | 1.74x |

One representative case rewrites a broadcast-heavy reduction into a GEMV-style streamed reduction. The main gain comes from removing unnecessary intermediate expansion and improving memory access efficiency. This is the clearest evidence that the framework can use profiler feedback to identify the actual bottleneck instead of only making surface-level code edits.

## Limitation

The main limitation is coverage. Only 131 of the 184 benchmark kernels make it into the main evaluation, so compilation failures, runtime failures, and unstable evaluation remain a significant part of the workflow.

| Limitation | Why It Matters |
| --- | --- |
| Limited end-to-end coverage | A sizable fraction of kernels never reach the main statistics, which weakens claims about robustness |
| Shallow iterative search | Later rounds often revisit semantically similar but performance-neutral variants |
| Weak optimization guidance | Profiling is informative, but it is not as directional as a gradient or analytical optimizer |
| Nontrivial runtime and API cost | 22 minutes and about $1.20 per kernel make large-scale deployment expensive |
| Long-tail performance variability | Large speedups exist, but many successful kernels still land in the modest 1.05x-1.5x range |

---

*Reading date: 2026-04*
*Note status: Completed*
