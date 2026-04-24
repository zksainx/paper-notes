# CUDA-L2: Surpassing cuBLAS Performance for Matrix Multiplication through Reinforcement Learning

<div class="paper-meta" markdown>

**Authors**: Songqiao Su, Xiaofei Sun, Xiaoya Li, Albert Wang, Jiwei Li, Chris Shum  
**Institution**: DeepReinforce Team  
**Conference**: arXiv  
**Year**: 2025  
**Paper Link**: [arXiv:2512.02551](https://arxiv.org/abs/2512.02551)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">HGEMM</span>
<span class="paper-tag">Reinforcement Learning</span>
<span class="paper-tag">cuBLAS</span>
</div>

## Background

CUDA-L2 targets one of the hardest and most commercially relevant CUDA kernels: half-precision GEMM on NVIDIA A100. Unlike prior benchmark-centered work that optimizes a wide variety of operators at one fixed shape per task, CUDA-L2 studies a production-like configuration space with 1,000 `(M, N, K)` combinations drawn from `{64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384}`. These shapes already cover common attention and FFN dimensions in open LLMs such as Qwen, Llama, and DeepSeek.

The paper asks a stronger question than most LLM-for-kernel studies: can RL-guided code generation beat NVIDIA's own highly tuned matrix libraries rather than only beating PyTorch or benchmark reference code? CUDA-L2 claims yes on A100 for HGEMM, including against `cuBLASLt-AutoTuning`, which already exhaustively tests up to 100 candidate algorithms per configuration.

**Key Takeaways**

- CUDA-L2 extends CUDA-L1 from general CUDA optimization to specialized HGEMM optimization over 1,000 shapes.
- The training pipeline adds continued pretraining, broader kernel RL, and a final HGEMM-focused RL stage with profiling context.
- On A100, CUDA-L2 reports average gains of `+11.4%` over `cuBLASLt-AutoTuning` in offline mode and `+15.9%` in server mode, while outperforming weaker baselines by larger margins.

## Methodology

CUDA-L2 builds on CUDA-L1 but makes the task much more specialized. The model is not only asked to generate fast CUDA code; it is trained to reason about tiling, pipeline depth, swizzling, data movement, and numerical correctness across a large HGEMM configuration space.

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Continued Pretraining | Diverse CUDA code from web sources, PyTorch, ATen, CUTLASS, and NVIDIA examples | CUDA-stronger foundation model | Expands beyond KernelBench-style operator coverage and adds newer CUDA knowledge |
| General Kernel RL | About 1K CUDA kernels paired with official reference implementations | RL-adapted general optimizer | Learns transferable CUDA optimization behavior before specializing to GEMM |
| HGEMM RL | HGEMM tasks over 1,000 `(M,N,K)` settings | Matmul-specialized policy | Tunes kernel structure and parameters specifically for half-precision GEMM |
| Contrastive Prompting | Prior kernel variants, scores, profiling context, retrieved references | Comparative reasoning plus new kernel | Reuses CUDA-L1's contrastive RL setup for performance-driven synthesis |
| Reward Function | Speedup, numerical deviation, code length | Scalar RL reward | Encourages fast, correct, and relatively concise kernels |

### From CUDA-L1 to CUDA-L2

CUDA-L2 identifies two main weaknesses in CUDA-L1 for matmul:

- KernelBench-style SFT data is too narrow for HGEMM-scale specialization.
- The model lacks newer CUDA ecosystem knowledge, especially around CUTLASS, CuTe, and more recent optimization idioms.

To address this, CUDA-L2 first performs continued pretraining on broader CUDA corpora. It then trains with RL in two steps: a general-kernel phase and a final HGEMM-only phase. This staged specialization keeps the model from overfitting too early to GEMM while still ending with a domain-specific optimizer.

### HGEMM RL and Reward Design

The final HGEMM reward combines three terms:

- execution speedup relative to the baseline
- penalty for numerical deviation from FP32 CPU ground truth
- penalty for excessive code length

The paper also feeds richer context into the model than CUDA-L1:

- NCU metrics such as memory throughput, occupancy, and cache efficiency
- retrieval-augmented documentation and code examples
- prior successful and unsuccessful variants for contrastive reasoning

Generated kernels are emitted as `.cu` files and compiled with `nvcc`, which allows raw CUDA C/C++, CuTe, CUTLASS templates, intrinsics, and inline PTX, while excluding Python DSLs like Triton.

### Key Design Choices

| Design Choice | Purpose | Effect |
| --- | --- | --- |
| Continued pretraining on broader CUDA sources | Fill missing CUDA knowledge | Makes the base model more capable on non-KernelBench kernels |
| Two-stage RL specialization | Learn general optimization before HGEMM specialization | Improves transfer while still targeting matmul performance |
| Offline and server evaluation modes | Measure both throughput and deployment-like latency behavior | Shows that improvements persist outside back-to-back execution |
| Strong correctness checks | Avoid false wins from numerical drift | Uses exact binary-input checks and baseline-bounded deviation |
| Profiling-aware context | Expose hardware bottlenecks directly to the model | Supports better choices for tiling, buffering, and swizzling |

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Target Kernel | HGEMM on A100 |
| Shape Space | 1,000 `(M,N,K)` configurations |
| Dimensions | `{64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384}` |
| Baselines | `torch.matmul`, `cuBLAS`, `cuBLASLt-heuristic`, `cuBLASLt-AutoTuning` |
| Layout Variants | `NN`, `TN`, and per-shape `max` of the two |
| Execution Modes | `offline` and `server` |
| Evaluation | minimum 30s timed run after 10s warmup with randomized order |

### Headline Results

| Baseline | Offline Gain | Server Gain | Interpretation |
| --- | --- | --- | --- |
| `torch.matmul` | `+22.0%` | `+28.7%` | Strongest gain versus common user-facing baseline |
| `cuBLAS-max` | `+19.2%` | `+26.0%` | Beats the best of cuBLAS `NN` and `TN` per shape |
| `cuBLASLt-heuristic-max` | `+16.8%` | `+22.4%` | Outperforms NVIDIA's heuristic-guided lower-level API |
| `cuBLASLt-AutoTuning-max` | `+11.4%` | `+15.9%` | Still wins against the strongest baseline tested |

The most important result is not the gain over `torch.matmul`; it is the margin over `cuBLASLt-AutoTuning`, because that baseline already searches up to 100 cuBLASLt candidate algorithms per configuration. Beating it on average across 1,000 shapes makes the paper materially stronger than benchmark papers that only compare against PyTorch or a single library default.

### Additional Deployment-Oriented Result

| Choice Rule | Offline Gain vs Baseline | Server Gain vs Baseline |
| --- | --- | --- |
| `max(CUDA-L2, torch.matmul)` | `+23.1%` | `+29.8%` |
| `max(CUDA-L2, cuBLAS-max)` | `+20.2%` | `+27.2%` |
| `max(CUDA-L2, cuBLASLt-heuristic-max)` | `+17.0%` | `+22.7%` |
| `max(CUDA-L2, cuBLASLt-AutoTuning-max)` | `+13.2%` | `+18.1%` |

This table reflects a more realistic deployment policy: if both CUDA-L2 and the baseline kernels are available, pick the faster one for each shape. The gains remain positive across all baselines, which means CUDA-L2 is not merely shifting wins around a few narrow configurations.

### Speedup Behavior and Discovered Techniques

| Observation | Main Finding |
| --- | --- |
| By problem size | Gains are largest on smaller matrices and shrink toward `1.0x` on very large matrices where the GPU is already near saturation |
| Abstraction choice | Raw WMMA is favored for small matrices; CuTe abstractions are favored for large matrices with more complex pipelining |
| Padding strategy | The model sometimes pads `M` with zeros to unlock better `BM` choices, e.g. using `BM=160` instead of divisor-friendly `128` |
| Advanced scheduling | CUDA-L2 discovers double-buffered register fragments, aggressive multi-step prefetching, optimized epilogues, and staggered A/B prefetch schedules |

One especially interesting example is padding-based tiling. For `(8192, 512, 2048)`, CUDA-L2 pads `M` from `8192` to `8320` so it can use `BM=160`. The paper reports that this gives `+15.2%` over `cublaslt-AutoTuning-TN`, while more standard `BM=128` nearly removes the gain and `BM=256` becomes substantially worse. This is a good example of RL exploiting a larger optimization space than hand-written divisibility heuristics usually consider.

## Limitation

CUDA-L2 is stronger than CUDA-L1, but the scope is narrower and the strongest claims are still tied to a specific hardware family and kernel class.

| Limitation | Why It Matters |
| --- | --- |
| Only A100 is evaluated in the main study | The paper states that other architectures are future work, so cross-GPU robustness is not yet demonstrated here |
| Focused only on HGEMM | The method may not transfer directly to broader operator families without another specialization stage |
| Gains shrink on large matrices | When workloads already saturate tensor-core throughput, the remaining headroom is limited |
| Heavy evaluation cost | Testing 1,000 shapes against multiple strong baselines and tuning modes is expensive |
| Library competition is narrow but deep | The result is compelling for matmul, but broader end-to-end model-level speedups are not measured |

---

*Reading date: 2026-04*
*Note status: Completed*
