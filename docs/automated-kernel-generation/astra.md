# Astra: A Multi-Agent System for GPU Kernel Performance Optimization

<div class="paper-meta" markdown>

**Authors**: Anne Ouyang, Azalia Mirhoseini, Ke Wang, Alex Aiken  
**Conference**: arXiv  
**Year**: 2025  
**Paper Link**: [arXiv:2509.07506](https://arxiv.org/abs/2509.07506)  
**GitHub**: [Anjiang-Wei/Astra](https://github.com/Anjiang-Wei/Astra)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Multi-Agent Systems</span>
<span class="paper-tag">CUDA Optimization</span>
<span class="paper-tag">SGLang</span>
</div>

## Background

Astra studies GPU kernel optimization in a setting closer to production deployment than benchmark-style code generation. Instead of translating PyTorch modules into CUDA, it starts from existing CUDA kernels already used inside SGLang and tries to improve their runtime while preserving exact behavior. This removes the translation problem and focuses the system entirely on the optimization loop.

The main claim is that kernel optimization is naturally multi-stage. A useful workflow needs representative test construction, correctness checking, runtime profiling, diagnosis, planning, and code rewriting. A single LLM can attempt all of these tasks, but the paper argues that performance optimization benefits from role specialization because failures in one stage, such as poor test inputs, can mislead the entire search process.

**Key Takeaways**

- Astra decomposes CUDA optimization into four specialized agent roles rather than using one monolithic coding agent.
- The system works directly on real kernels from SGLang, then reintegrates the optimized versions back into the framework.
- On three production kernels, Astra reaches 1.32x average speedup with zero-shot prompting using `o4-mini`, outperforming a single-agent baseline at 1.08x.

## Methodology

Astra is an iterative multi-agent pipeline for optimizing an existing CUDA kernel `S` into a faster implementation `S'` while preserving correctness. The paper defines correctness through agreement with the baseline kernel on a finite test suite, allowing a tolerance for floating-point deviations, and reports performance using geometric mean speedup across representative tensor shapes.

The system runs for `R = 5` rounds. It first constructs an initial test suite and profiles the baseline kernel. Each round then cycles through planning, code generation, correctness testing, and profiling, while logging the candidate code, correctness status, and measured performance. A manual pre-processing step extracts stand-alone kernels from SGLang before optimization, and a manual post-processing step patches optimized kernels back into SGLang for final validation against the original framework behavior.

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Testing Agent | Baseline kernel or candidate kernel | Test suite and correctness result | Builds representative tests and rejects functionally incorrect candidates |
| Profiling Agent | Candidate kernel plus test inputs | Runtime measurements | Measures execution time and provides performance feedback |
| Planning Agent | Correctness and profiling feedback | Optimization suggestions | Decides what transformation to try next |
| Coding Agent | Current kernel plus plan | Revised CUDA kernel | Applies the proposed rewrite |
| Optimization Log | Per-round artifacts | `(round, code, correctness, performance)` records | Tracks search history across rounds |

### Pipeline Structure

| Stage | What Happens | Why It Matters |
| --- | --- | --- |
| Pre-processing | Manually extract and simplify SGLang kernels into stand-alone CUDA programs | Makes the optimization target executable outside framework internals |
| Initialization | Construct initial tests and baseline profiling results | Establishes a correctness and runtime reference point |
| Iterative Optimization | Planning -> coding -> testing -> profiling for 5 rounds | Lets agents refine kernels using both correctness and performance feedback |
| Post-processing | Monkey-patch optimized kernels back into SGLang and validate against original behavior | Confirms that improvements survive framework reintegration |

### Key Design Choices

- Astra uses role specialization instead of a single general-purpose agent.
- The target is optimization of existing kernels, not end-to-end synthesis from Python or PyTorch.
- Final correctness is checked with manually designed evaluation tests rather than only the testing agent's generated suite.
- Speedup is reported as a geometric mean across realistic tensor shapes taken from deployed LLMs.

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Framework Target | SGLang |
| Kernels | `merge_attn_states_lse`, `fused_add_rmsnorm`, `silu_and_mul` |
| Model | OpenAI `o4-mini` |
| Agent Framework | OpenAI Agents SDK |
| Hardware | NVIDIA H100 |
| Optimization Rounds | 5 |
| Runtime Measurement | 20 warm-up runs + 100 measured repetitions per shape |
| Shape Selection | Common dimensions from LLaMA-7B, 13B, and 70B style workloads |
| Metrics | Correctness and geometric-mean speedup |

### Headline Results

| Kernel | Time-Base (us) | Time-Opt. (us) | Speedup | Correct |
| --- | --- | --- | --- | --- |
| `merge_attn_states_lse` | 31.4 | 24.9 | 1.26x | Yes |
| `fused_add_rmsnorm` | 41.3 | 33.1 | 1.25x | Yes |
| `silu_and_mul` | 20.1 | 13.8 | 1.46x | Yes |
| **Average** | **30.9** | **23.9** | **1.32x** | **Yes** |

The gains are modest but credible because they are measured on existing production kernels rather than on synthetic translation tasks. The best improvement appears on `silu_and_mul`, while all three kernels remain correct after optimization and framework-level reintegration.

### Multi-Agent vs. Single-Agent

| Kernel | Speedup - Single Agent | Speedup - Multi-Agent |
| --- | --- | --- |
| `merge_attn_states_lse` | 0.73x | 1.26x |
| `fused_add_rmsnorm` | 1.18x | 1.25x |
| `silu_and_mul` | 1.48x | 1.46x |
| **Average** | **1.08x** | **1.32x** |

The comparison isolates the value of specialization. The largest difference appears on the most complex kernel, `merge_attn_states_lse`, where the single-agent setup regresses to 0.73x because poor test construction distorts profiling feedback. On the simplest kernel, the two approaches are nearly tied, which suggests that role decomposition matters most when the optimization workflow itself becomes brittle.

### Case Study Signals

| Kernel | Main Optimization Pattern | Effect |
| --- | --- | --- |
| `merge_attn_states_lse` | Hoist loop-invariant exponential and normalization work out of the hot inner loop | Reduces repeated expensive operations and improves throughput |
| `fused_add_rmsnorm` | Replace shared-memory-only reduction with warp-level intrinsics plus short shared-memory phase | Cuts synchronization overhead and memory traffic |
| `silu_and_mul` | Use `__half2` vectorized loads and fast math intrinsics such as `__expf` and reciprocal-multiply | Improves both memory bandwidth and arithmetic efficiency |

The case studies show that Astra is not only tuning launch parameters. It performs structural rewrites, introduces warp intrinsics, changes memory-access granularity, and substitutes lower-latency device intrinsics. The paper also reports shape-by-shape speedups, where some inputs reach 1.57x while others drop to 1.00x, so the average gains are not uniformly distributed across all workloads.

## Limitation

The paper is explicit that Astra is an early system rather than a fully automated optimizer. Its current evaluation is small, and the surrounding workflow still requires substantial manual engineering.

| Limitation | Why It Matters |
| --- | --- |
| Only three kernels are evaluated | The evidence is promising but too narrow to establish broad generalization |
| The workflow is SGLang-specific | It is unclear how much of the pipeline transfers directly to other serving or training stacks |
| Pre-processing and post-processing are manual | Extracting stand-alone kernels and reintegrating them back into the framework remains a major bottleneck |
| Final correctness uses manually designed tests | This improves confidence, but also means the system is not yet fully autonomous end to end |
| Shape robustness is uneven | Some shapes see large gains, while others are neutral, so the optimizer is not shape-specialized or uniformly strong |

---

*Reading date: 2026-04*
*Note status: Completed*
