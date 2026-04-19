# PEAK: A Performance Engineering AI-Assistant for GPU Kernels Powered by Natural Language Transformations

<div class="paper-meta" markdown>

**Authors**: Muhammad Usman Tariq, Abhinav Jangda, Angelica Moreira, Madan Musuvathi, Tyler Sorensen  
**Institution**: Stanford University, Microsoft Research Redmond, University of California Santa Cruz  
**Conference**: arXiv  
**Year**: 2025  
**Paper Link**: [arXiv:2512.19018](https://arxiv.org/abs/2512.19018)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">GPU Kernels</span>
<span class="paper-tag">Natural Transformations</span>
<span class="paper-tag">Cross-Backend</span>
</div>

## Background

PEAK is built around a simple but useful observation: many GPU kernel optimizations are easier to describe in natural language than to encode as rigid compiler passes. Instead of treating kernel generation as a monolithic one-shot synthesis problem, PEAK treats optimization as a sequence of explicit transformations that can be interpreted, applied, validated, and iterated on by an LLM-assisted system.

The paper focuses on matrix multiplication because it is both performance-critical and structurally rich enough to exercise tiling, tensor core usage, pipelining, caching, and backend-specific tuning. The broader goal is not only to build a competitive MatMul optimizer, but also to expose how LLMs behave when asked to carry out realistic low-level kernel transformations across different GPU ecosystems.

**Key Takeaways**

- PEAK frames GPU kernel optimization as interpretable natural-language transformations rather than opaque end-to-end generation.
- The same framework is instantiated across CUDA, HIP, and HLSL, which highlights both portability and backend-specific specialization.
- For MatMul, the resulting kernels are competitive with vendor libraries on NVIDIA and AMD, and reach hardware-level documented FLOPS on HLSL where a standard vendor library is unavailable.

## Methodology

PEAK is a modular optimization framework that applies natural-language transformation specifications to a kernel context, then validates correctness and evaluates performance after each step. The core design aims to preserve human interpretability: each optimization remains visible as a named transformation with explicit intent, instead of disappearing inside a black-box search loop.

The kernel context is broader than kernel source alone. It includes the device kernel, host launcher code, input specification, and a set of performance tuning parameters. This gives the system enough context to modify both implementation structure and evaluation setup while staying grounded in executable kernels.

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Natural Transformations | Kernel context plus a natural-language optimization instruction | Updated kernel and host code | Applies an interpretable optimization step such as tiling, tensor-core use, or pipelining |
| Kernel Context | Kernel, launcher, input sizes, tuning parameters | Structured optimization state | Provides enough surrounding context for low-level edits and backend-specific execution |
| Correctness Validators | Candidate kernel and test inputs | Pass / fail feedback | Checks semantic correctness after each transformation |
| Performance Evaluators | Candidate kernel and tuning space | Runtime and throughput measurements | Searches or enumerates promising performance settings |
| Performance Workflow | Seed kernel plus ordered transformation sequence | Final optimized implementation | Orchestrates iterative optimization, validation, checkpointing, and rollback |

### Representative Natural Transformations

| Transformation | Purpose | Passes / LLM Calls | Tuning Parameters |
| --- | --- | --- | --- |
| Refactor | Rewrite complex accesses into reusable macros and cleaner primitives | 2 / 3 | None |
| TB-Tiling | Introduce thread-block tiling | 6 / 9 | `TILE_K_SIZE` |
| Warp-Tiling | Map work to warp-level structure | 1 / 1 | `WARP_X_DIM` |
| Thread-Tiling | Increase per-thread reuse | 3 / 3 | `TRD_X_DIM`, `TRD_Y_DIM` |
| Tensor-Core | Switch to tensor-core execution paths | 1 / 1 | None |
| Split-K | Parallelize along the reduction dimension | 1 / 3 | `K_SPLITS` |
| Pipelining | Overlap memory movement and compute | 1 / 2 | `NUM_STAGES` |

### Key Design Choices

- The system is backend-agnostic at the framework level and keeps backend-specific logic mostly in drivers and a small number of specialized transformations.
- Natural transformations are lightweight to write and revise, which makes them easier to extend than compiler passes or RL-heavy search policies.
- Validation and performance search are first-class parts of the workflow rather than post-processing steps.
- The paper studies a semi-autonomous setting with human-designed transformation sequences, but the same space could be searched automatically or proposed by an agent.

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Task | Matrix multiplication kernels |
| Backends | CUDA, HIP, HLSL |
| Devices | NVIDIA A6000, AMD MI200, Qualcomm Adreno X1-85 |
| Precisions | fp32 and fp16 where supported |
| Input Sizes | 2048 and 4096 square matrices |
| Optimization Target | Performance relative to naive baseline and hardware/library ceilings |
| Evaluation Pattern | Apply ordered transformation sequences, validate correctness, then evaluate performance |

### Headline Results

| Device | Backend | Precision | Baseline Speedup | % of Max FLOPS | # Transformations |
| --- | --- | --- | --- | --- | --- |
| NVIDIA A6000 | CUDA | fp32 | 9.36x / 10.36x | 95.0% / 91.6% | 11 |
| NVIDIA A6000 | CUDA | fp16 | 41.06x / 45.34x | 99.4% / 89.5% | 9 |
| AMD MI200 | HIP | fp32 | 24.39x / 25.21x | 91.9% / 86.2% | 8 |
| AMD MI200 | HIP | fp16 | 36.14x / 39.56x | 48.2% / 37.2% | 5 |
| Qualcomm Adreno X1-85 | HLSL | similar | 4.16x / 14.71x | 107.3% / 109.8% | 3 |
| **Average** |  |  | **19.52x** | **78.26%** | **7.3** |

The core result is that PEAK reaches competitive MatMul performance without relying on an opaque RL pipeline. NVIDIA results often exceed 90% of peak or library-level throughput, AMD remains strong for fp32, and HLSL reaches the documented hardware FLOPS despite the lack of a standard BLAS-style library. That cross-backend portability is one of the paper's strongest claims.

### Performance Workflow Examples

| Device | Precision | Transformation Sequence | # Transforms |
| --- | --- | --- | --- |
| A6000 | fp32 | Refactor -> TB-Tiling -> Thread-Tiling -> Thread-Cache -> Transpose -> Thread-Chunk -> Split-K -> Pipelining -> Register-Staging -> Offset -> Block-Swizzle | 11 |
| A6000 | fp16 | Refactor -> TB-Tiling -> Warp-Tiling -> Tensor-Core -> Tensor-Tiling -> Pipelining -> Register-Staging -> Offset -> Block-Swizzle | 9 |
| MI200 | fp32 | Refactor -> TB-Tiling -> Warp-Tiling -> Tensor-Core -> Tensor-Tiling | 5 |
| MI200 | fp16 | Refactor -> TB-Tiling -> Warp-Tiling -> Tensor-Core -> Tensor-Tiling -> Offset -> Block-Swizzle -> Pipelining | 8 |
| X1-85 | both | Refactor -> TB-Tiling -> Thread-Tiling | 3 |

These sequences show how PEAK behaves more like a performance-engineering workflow than a single optimizer call. Some transformations are common across backends, while others diverge because of hardware-specific constraints such as shared-memory limits or tensor-core APIs.

### End-to-End Optimization Cost

| Workflow | Validation Time | Transformation Time | Performance Search Time | Total |
| --- | --- | --- | --- | --- |
| A6000 fp32 | 825 s (3.91%) | 1285 s (6.09%) | 18995 s (89.99%) | 21105 s |
| A6000 fp16 | 1124 s (1.55%) | 257 s (6.77%) | 15226 s (91.68%) | 16607 s |

The runtime breakdown shows that transformation generation itself is not the main bottleneck. Performance evaluation dominates the workflow, which suggests that future improvements depend less on faster prompting and more on better search-space pruning or stronger performance estimators.

## Limitation

PEAK trades full autonomy for interpretability and control. That makes the system easier to inspect and debug, but it also means the reported results still depend on curated transformation sequences for the case study.

| Limitation | Why It Matters |
| --- | --- |
| Human-seeded transformation workflows | The study uses expert-designed sequences, so autonomy is demonstrated more as a future direction than a fully solved capability |
| Case study centered on MatMul | Results are strong on a very important kernel, but generality across broader kernel families is not yet established at the same depth |
| Performance search dominates runtime | More than 90% of end-to-end time can go into evaluation and tuning, which limits scalability |
| Mid-workflow optimization can be unstable | The paper notes that performance does not improve monotonically across transformation sequences |
| Backend-specific adaptation still exists | The framework is portable, but some transformations and drivers still require backend-specific knowledge |

---

*Reading date: 2026-04*
*Note status: Completed*
