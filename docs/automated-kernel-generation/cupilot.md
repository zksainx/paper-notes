# cuPilot: A Strategy-Coordinated Multi-agent Framework for CUDA Kernel Evolution

<div class="paper-meta" markdown>

**Authors**: Jinwu Chen, Qidie Wu, Bin Li, Lin Ma, Xin Si, Yang Hu, Shouyi Yin, Jun Yang  
**Institution**: Southeast University, Tsinghua University, Tsing Micro, National Center of Technology Innovation for EDA  
**Conference**: arXiv  
**Year**: 2025  
**Paper Link**: [arXiv:2512.16465](https://arxiv.org/abs/2512.16465)  
**GitHub**: [champloo2878/cuPilot-Kernels](https://github.com/champloo2878/cuPilot-Kernels.git)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Multi-Agent Systems</span>
<span class="paper-tag">CUDA Optimization</span>
<span class="paper-tag">Evolutionary Search</span>
</div>

## Background

cuPilot targets a specific weakness in LLM-based CUDA optimization: current agentic evolution frameworks often operate directly on low-level kernel code, while the actual optimization logic lives at a higher semantic level. This creates three mismatches. Code-level crossover forces the model to infer and recombine strategies from raw source; fitness signals expressed only as runtime metrics are weakly aligned with code edits; and initializing the population from a small set of kernels gives poor coverage of the strategy space. The result is that earlier systems often keep correctness but fail to accumulate sophisticated optimizations across generations.

The paper introduces `strategy` as an intermediate representation between the evolutionary algorithm and CUDA source. cuPilot then uses three mechanisms to make this representation actionable: a Strategy-Coordinated Evolution algorithm, roofline-guided prompting, and strategy-level population initialization with historical retrieval. The goal is not just to generate valid kernels, but to evolve kernels whose strategies can be recombined and refined more reliably over time.

**Key Takeaways**

- cuPilot replaces direct kernel-to-kernel crossover with a two-step strategy-level evolution process.
- Roofline-guided prompts convert generic performance optimization into bounded compute-focused or memory-focused strategy search.
- On 100 KernelBench Level-1 kernels, cuPilot reports `3.09x` average speedup over PyTorch and clearly outperforms AI CUDA Engineer under the same A100-based evaluation setup.

## Methodology

cuPilot organizes kernel evolution around explicit strategy semantics rather than raw code mutation alone. The framework partitions work into three main roles: an SCE Manager that controls population evolution, a Strategy Translator that maps strategies into candidate CUDA kernels, and a Kernel Revisor that repairs syntax, functional errors, and performance issues using profiling feedback. This design shortens the reasoning path for the LLM: instead of asking the model to fuse two complicated kernels directly, cuPilot first recombines their strategies and only then synthesizes code.

The framework adds two more sources of structure. Roofline-guided prompting uses hardware bottleneck analysis to bias strategies toward compute or memory optimization depending on the kernel’s position on the roofline model. Strategy-level population initialization uses both short-term history and a RAG-style external strategy pool so that initial candidates cover the strategy space better than sparse code-only seeding. Together, these mechanisms aim to increase convergence quality and reduce premature collapse to shallow optimizations.

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| SCE Manager | Current population, fitness signals, generation state | Selected parents and next-generation strategy plans | Runs elitism, tournament selection, and evolution scheduling |
| Strategy Translator | Initial kernel plus explicit optimization strategy | Candidate CUDA kernel | Realizes strategy semantics as concrete code |
| Kernel Revisor | Candidate kernel, compile/runtime/profile feedback | Repaired or refined kernel | Fixes syntax, functionality, and performance issues |
| Roofline Analyzer | Profiling metrics and hardware limits | Compute-bound / memory-bound diagnosis | Guides prompts toward the dominant bottleneck |
| Strategy Memory / RAG | Historical kernel-strategy-performance tuples | Retrieved seed strategies | Improves initialization and cross-task reuse |

### Core Mechanisms

| Mechanism | What It Changes | Why It Matters |
| --- | --- | --- |
| Strategy-Coordinated Evolution | Crosses strategies first, then translates to kernels | Avoids asking the LLM to decode and recombine complex kernels directly |
| Roofline-Guided Prompting | Conditions prompts on compute vs memory bottlenecks | Focuses optimization on the most relevant hardware resource |
| Strategy-Level Initialization | Seeds evolution from retrieved strategies, not only raw kernels | Expands strategy coverage and improves convergence |
| Kernel Revision Loop | Revises generated kernels using syntax, function, and profiling feedback | Preserves viable strategies while repairing implementation failures |

### Evolution Pipeline

| Stage | Main Action | Effect |
| --- | --- | --- |
| Strategy Initialization | Build an initial strategy population for a kernel | Creates a broader starting search space |
| Strategy Application | Translate strategies into CUDA candidates | Produces explicit code realizations of semantic plans |
| Kernel Revision | Repair and profile generated kernels | Filters invalid candidates and improves fitness |
| Strategy Alignment | Update strategy descriptions using optimized kernels | Keeps strategy semantics consistent with actual code |
| Selection and Crossover | Apply elitism and tournament selection at strategy level | Preserves strong individuals while recombining useful ideas |

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Hardware | Server with 8x NVIDIA A100 GPUs |
| CUDA Version | 12.4 |
| Benchmark | KernelBench Level-1 |
| Main Evaluation Set | 100 kernels |
| Profiling Tool | Nsight Compute CLI |
| Runtime Measurement | 10 warm-up runs + 50 timed runs with random inputs |
| GPU Clock | Locked at 1410 MHz |
| Models | DeepSeek-R1, Gemini-2.5 Pro |
| Main External Baseline | AI CUDA Engineer |

### Headline Results

| Setting | Result |
| --- | --- |
| Overall average speedup over PyTorch | `3.09x` |
| GEMM kernels | `4.06x` |
| CONV kernels | `1.18x` |
| Activation / Normalization kernels | `4.16x` |
| Pooling kernels | `3.85x` |
| Loss kernels | `6.87x` |

The aggregate result is strong for a code-generation system that targets open CUDA kernels rather than wrapped vendor libraries. The largest gains appear on GEMM and other operator families where multiple optimizations can be stacked, while CONV is more modest. The paper also notes that cuPilot’s results were produced with less than three hours of per-kernel runtime for each epoch, which matters because these systems can otherwise become prohibitively slow.

### Ablation Results

| Ablation Target | Setup | Main Finding |
| --- | --- | --- |
| Roofline-Guided Prompting | 4 representative kernels, DeepSeek-R1 and Gemini-2.5 Pro, 2 epochs, population size 50, 3 repeats | Average latency drops by `44.2%` after two epochs with roofline guidance |
| Strategy-Level Population Initialization with RAG | Same 4 kernels, compare with vs without external strategy pool | Average latency drops by `54.1%` after the first epoch with RAG-assisted initialization |

These ablations support the paper’s claim that cuPilot is not winning from one trick alone. Roofline guidance improves the direction of each optimization step, while retrieval-enhanced initialization gives the evolution loop a stronger starting population. The paper also reports that even without roofline guidance, cuPilot still outperforms AI CUDA Engineer on the studied kernels, which suggests that the strategy-level representation itself is already doing substantial work.

### GEMM Case Study

| Aspect | cuPilot | AI CUDA Engineer |
| --- | --- | --- |
| Compute optimization | Uses Tensor Cores and more advanced compute strategies | Mostly basic strategies |
| Memory optimization | Uses padding, layout swizzling, thread-block swizzling, async copy | More limited vectorized-access-style improvements |
| Pipeline optimization | Applies double buffering or multi-stage pipelines | Weaker pipeline sophistication |
| Library reliance | Generates authentic optimized kernels | Reported GEMM examples can act as cuBLAS wrappers |

The GEMM study is the clearest evidence that cuPilot is learning more than superficial rewrites. According to the paper, cuPilot combines Tensor Core use, bank-conflict reduction, cache-locality improvements, and staged pipelines in a way that increases both compute utilization and memory throughput. By contrast, the baseline often relies on simpler tiling or vectorization, and in some GEMM cases effectively wraps existing library code instead of solving the kernel-optimization problem directly.

## Limitation

The reported gains are promising, but the evidence base is narrower than the headline numbers may initially suggest.

| Limitation | Why It Matters |
| --- | --- |
| Main benchmark evaluation covers only 100 Level-1 KernelBench kernels | It does not establish the same level of evidence for harder fused or model-level workloads |
| Results are mostly aggregated by operator family | The paper gives limited per-kernel breakdowns, so robustness across the full set is harder to judge |
| Comparison focuses heavily on AI CUDA Engineer | The paper does not provide equally detailed comparisons against a wider range of recent kernel agents |
| Institution metadata is partially recoverable from arXiv metadata only | Affiliation granularity appears incomplete at the author-by-author level |
| Runtime cost is still nontrivial | Under-three-hours-per-kernel-per-epoch is better than many long-horizon agents, but still expensive for broad deployment |

Another practical limitation is that the strongest claims are about strategy representation rather than rigorous benchmarking methodology. The paper makes a convincing systems argument for strategy-level crossover, but the evaluation would be stronger with more extensive coverage beyond Level-1, more direct reporting on correctness failure rates, and broader baseline comparisons.

---

*Reading date: 2026-04*
*Note status: Completed*
