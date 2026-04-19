# CUDA-LLM: LLMs Can Write Efficient CUDA Kernels

<div class="paper-meta" markdown>

**Authors**: Wentao Chen, Jiace Zhu, Qi Fan, Yehan Ma, An Zou  
**Institution**: Shanghai Jiao Tong University  
**Conference**: arXiv  
**Year**: 2025  
**Paper Link**: [arXiv:2506.09092](https://arxiv.org/abs/2506.09092)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">CUDA</span>
<span class="paper-tag">Performance Reinforcement</span>
<span class="paper-tag">Kernel Optimization</span>
</div>

## Background

CUDA-LLM studies whether an LLM can do more than emit syntactically plausible CUDA kernels. The core challenge is that CUDA programming is hardware-specific, architecture-aware, and strongly performance-sensitive. Correct indexing and functional validity are only the first hurdle; useful kernels also need coalesced memory access, balanced resource usage, and low synchronization overhead on real GPUs.

The paper addresses this gap with a framework called Feature Search and Reinforcement (FSR). Instead of treating kernel generation as a one-shot prompting problem, FSR uses actual compilation outcomes, functional test results, and runtime latency measurements to iteratively steer the model toward valid and faster CUDA code.

**Key Takeaways**

- FSR turns CUDA generation into an iterative repair-and-optimization loop grounded in real execution feedback.
- The framework improves both correctness and performance relative to direct LLM generation.
- On a 20-task benchmark suite, CUDA-LLM solves all tasks on both test GPUs and reports speedups up to 179x over the human-written baseline.

## Methodology

CUDA-LLM is centered on FSR, a loop that evaluates candidate kernels along three dimensions: compilation, functional correctness, and execution performance. The design separates these checks into explicit feature functions so the system can reject invalid kernels early and only spend runtime profiling budget on code that is already executable and correct.

The workflow begins with an initial prompt that asks the model to produce multiple candidate CUDA kernels. If none of them both compiles and passes the functional validator, FSR refines the prompt using the candidate code, the observed error messages, and the interaction history, then generates a new batch. Once at least one valid kernel appears, the system profiles those kernels on real hardware, keeps the fastest one, and uses that best candidate plus its optimization history as the starting point for the next refinement round.

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Initial Prompting | Task description and kernel objective | `N` candidate CUDA kernels | Produces the first batch of implementations |
| Compilation Verifier | Candidate kernel | Compile pass / error log | Filters out syntactically or toolchain-invalid kernels |
| Function Validator | Compiled kernel and test inputs | Correct / incorrect result | Ensures semantic correctness against reference outputs |
| Performance Profiler | Valid kernels on target GPU | Execution latency and speed ranking | Measures runtime efficiency on real hardware |
| Prompt Refinement Loop | Candidate code, errors, history, best kernel | New refined prompt | Feeds back failure and performance signals to improve the next batch |

### Feature Functions in FSR

| Feature Function | What It Checks | Why It Matters |
| --- | --- | --- |
| Compilation Verifier | Whether the kernel builds successfully | Prevents invalid CUDA from entering later stages |
| Function Validator | Whether outputs match reference behavior | Guards against semantically wrong but compilable kernels |
| Performance Profiler | Actual kernel runtime on GPU | Makes optimization target concrete and hardware-aware |

### Key Design Choices

- Correctness and performance are optimized jointly, but in stages: compile first, then validate outputs, then profile.
- Prompt refinement is driven by concrete evidence such as compiler diagnostics and failing behavior, not only by free-form model self-reflection.
- The benchmark includes varied CUDA tasks, which makes the method less dependent on a single kernel family.
- The paper uses DeepSeek-V3-0324 as the backbone model for code reasoning and generation.

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Benchmark Suite | 20 CUDA kernel tasks |
| Task Sources | NVIDIA CUDA Samples, LeetGPU, KernelBench |
| Evaluation Metrics | Compilation + function correctness, runtime latency, speedup over baseline |
| Backbone Model | DeepSeek-V3-0324 |
| Edge GPU | NVIDIA GeForce GTX 1660 SUPER |
| Server GPU | NVIDIA GeForce RTX 3090 Ti |
| Workload Style | Multiple input sizes per task to test robustness and scalability |

### Correctness Results

| GPU | Direct LLM Generation | CUDA-LLM with FSR |
| --- | --- | --- |
| GTX 1660 SUPER | Fails several tasks under pass@5 generation | Passes all 20 tasks |
| RTX 3090 Ti | Fails several tasks under pass@5 generation | Passes all 20 tasks |

The correctness gap is one of the strongest results in the paper. Direct generation can produce kernels that compile or look plausible but still fail functional testing. FSR closes that gap by repeatedly feeding compilation and validation failures back into the generation loop until a valid kernel is found.

### Headline Performance Results

| Result Type | Reported Gain |
| --- | --- |
| Best observed speedup | 179x |
| Matrix Transpose on RTX 3090 Ti | 104x |
| Dot Product on RTX 3090 Ti | 102x |
| Monte Carlo Integration on RTX 3090 Ti | 179x |
| Matrix Copy on GTX 1660 SUPER | 89.5x |
| Reduction on GTX 1660 SUPER | 73.3x |
| Typical gain range on RTX 3090 Ti | 5.95x to 68.3x |
| Typical gain range on GTX 1660 SUPER | 6.06x to 60.0x |

The performance gains are uneven across tasks, which is expected for a heterogeneous benchmark. Some kernels benefit from dramatic improvements in memory access structure or warp-level cooperation, while others see more modest but still meaningful acceleration. The paper emphasizes that the framework improves execution efficiency on nearly all tasks rather than only on a handful of cherry-picked cases.

### Where the Speedups Come From

| Task Pattern | Optimization Effect |
| --- | --- |
| Matrix Transpose | Coalesced reads and writes, reduced transactions, loop unrolling, lighter address arithmetic |
| Reduction / Dot Product / Monte Carlo Integration | Better use of warp-level primitives and lower synchronization overhead |
| General Memory-Bound Kernels | Improved memory behavior and reduced wasted accesses |

The qualitative analysis suggests that FSR is not merely repairing syntax. It often pushes the model toward standard high-value CUDA optimizations such as coalescing, warp-level communication, and more efficient thread coordination.

## Limitation

CUDA-LLM shows that feedback-driven prompting can make LLM-generated CUDA both correct and fast, but the paper still leaves several open constraints around generality and evaluation scope.

| Limitation | Why It Matters |
| --- | --- |
| CUDA-specific focus | The framework is evaluated only on CUDA, so portability beyond NVIDIA-style backends is not demonstrated here |
| Benchmark breadth vs. depth | The study spans 20 tasks, but the paper does not provide the same level of deep per-kernel analysis for every task |
| Large performance variance across tasks | Some kernels gain dramatically, while others improve less, which suggests uneven optimization reliability |
| Heavy dependence on runtime evaluation | Real execution latency is essential to the loop, so optimization cost can grow with search depth and test diversity |
| No broader cross-backend comparison | Unlike later systems that test CUDA, HIP, and other backends, this work does not study transfer across programming ecosystems |

---

*Reading date: 2026-04*
*Note status: Completed*
