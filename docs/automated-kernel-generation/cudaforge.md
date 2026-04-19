# CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization

<div class="paper-meta" markdown>

**Authors**: Zijian Zhang, Rong Wang, Shiyang Li, Yuebo Luo, Mingyi Hong, Caiwen Ding  
**Institution**: University of Minnesota, Twin Cities  
**Conference**: arXiv  
**Year**: 2025  
**Paper Link**: [arXiv:2511.01884](https://arxiv.org/abs/2511.01884)  
**GitHub**: [OptimAI-Lab/CudaForge](https://github.com/OptimAI-Lab/CudaForge)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Hardware Feedback</span>
<span class="paper-tag">CUDA Optimization</span>
<span class="paper-tag">Multi-Agent Systems</span>
</div>

## Background

CudaForge addresses a practical gap in LLM-based CUDA optimization. Existing approaches either rely on direct one-shot generation, which often fails on correctness or performance, or use reinforcement learning and complex agentic pipelines that remain expensive and difficult to generalize. The paper argues that one key missing ingredient is explicit hardware feedback. Human CUDA engineers do not optimize blindly: they iterate through correctness testing, runtime profiling, bottleneck diagnosis, and targeted revision. CudaForge tries to reproduce this loop at inference time without any additional training.

The system is built around a simple claim: a lightweight training-free workflow can outperform heavier alternatives if it separates generation from diagnosis and feeds the diagnosis with the right hardware signals. Instead of asking one model to both write and critique kernels, CudaForge splits the work between a Coder and a Judge. The Judge uses GPU specifications and carefully filtered Nsight Compute metrics to identify one critical bottleneck at a time, then sends focused correction or optimization guidance back to the Coder.

**Key Takeaways**

- CudaForge is a two-agent CUDA optimization workflow that combines correctness feedback with hardware-aware optimization feedback.
- The framework does not use the full NCU metric dump; it selects a compact 24-metric subset and then narrows attention to 3-4 critical metrics per round.
- On full KernelBench Levels 1-3, CudaForge reports `97.6%` correctness and `1.68x` average speedup over PyTorch, while remaining far cheaper than prior agentic pipelines.

## Methodology

CudaForge iterates through a Coder-Judge loop. The Coder produces a candidate CUDA kernel from the task description, previous kernel, and Judge feedback. The candidate is then compiled and executed against correctness tests. If it fails, the Judge enters correction mode and explains the likely bug. If it passes, the system profiles the kernel with Nsight Compute and the Judge enters optimization mode, using both NCU metrics and static GPU specifications to identify the dominant bottleneck and suggest the next targeted improvement.

The design is deliberately narrow and practical. The Coder does not see the full conversation history, only the previous kernel, task description, and latest feedback. The Judge also operates round-by-round and emits structured JSON feedback. The paper argues that this lightweight memory policy reduces hallucinations, lowers API cost, and keeps the optimization loop focused. A second important design choice is metric selection: instead of passing the full set of NCU metrics, the workflow builds an offline-selected subset of 24 metrics that are strongly correlated with runtime across representative tasks, then asks the Judge to focus on just a few salient ones in each round.

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Coder | Task description, previous kernel, Judge feedback | Candidate CUDA kernel | Generates or refines kernels round by round |
| Correctness Tester | Candidate kernel and reference task | Compile / run status and numerical result | Filters invalid kernels before optimization |
| Judge | Candidate kernel, runtime signals, hardware feedback | JSON correction or optimization guidance | Diagnoses errors and bottlenecks |
| Hardware Feedback Module | GPU specs plus selected NCU metrics | Bottleneck evidence | Grounds the Judge in actual device behavior |
| Selection Logic | All correct candidates across rounds | Best correct kernel | Chooses the most efficient valid result |

### Key Design Choices

- Separate generation and diagnosis into two agents instead of using one model to self-refine.
- Use correction mode and optimization mode separately, so correctness repair is not mixed with performance tuning.
- Keep only lightweight round-local memory for both agents.
- Select a compact 24-metric NCU subset offline instead of dumping the full profiler output into the prompt.
- Force the Judge to focus on one dominant bottleneck per round, similar to an expert performance-debugging workflow.

### Optimization Loop

| Stage | What Happens | Why It Matters |
| --- | --- | --- |
| Initial Generation | Coder writes a first CUDA candidate from the task prompt and one-shot example | Produces a concrete kernel to test |
| Compilation and Execution | The system checks syntax and output equivalence to PyTorch with tolerance `1e-4` | Distinguishes broken kernels from optimizable ones |
| Correction Feedback | Judge explains compile or correctness failures | Stabilizes correctness before optimization |
| NCU Profiling | Correct kernels are profiled with selected metrics | Reveals concrete hardware bottlenecks |
| Optimization Feedback | Judge identifies one critical bottleneck and suggests one focused change | Avoids blind exploration |
| Iterative Refinement | Coder rewrites the kernel using the latest feedback | Accumulates performance gains over rounds |

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Benchmark | KernelBench Levels 1-3 |
| Full Benchmark Size | 250 tasks |
| Stratified Subset | 25 tasks (`10/10/5` across Levels 1/2/3) |
| Default Base Model | OpenAI-o3 for both Coder and Judge |
| Default Max Rounds | `N = 10` |
| Main Hardware for Results | Quadro RTX 6000 |
| Comparison Hardware | H200, A100, RTX 4090, RTX 3090 |
| Correctness Criterion | Compile + execution pass with numerical tolerance `1e-4` |
| Main Baselines | OpenAI-o3, o3-self-refine, o3-correction, o3-optimization, Kevin-32B, Agentic Baseline |

### Headline Results

| Method / Setting | Correct | Median | 75th Pctl | Avg. Perf | Fast1 |
| --- | --- | --- | --- | --- | --- |
| OpenAI-o3 | 57.6% | 0.390 | 1.014 | 0.680 | 31.6% |
| o3-self-refine | 90.8% | 1.012 | 1.209 | 1.107 | 55.2% |
| o3-correction | 97.6% | 1.031 | 1.238 | 1.222 | 59.6% |
| o3-optimization | 88.4% | 1.061 | 1.483 | 1.509 | 64.0% |
| CudaForge | 97.6% | 1.107 | 1.592 | 1.677 | 70.8% |

On the full 250-task benchmark, CudaForge is stronger than both the raw base model and every internal ablation. The comparison is especially useful because the ablations isolate the value of each design choice. Correction-only feedback preserves the same correctness rate as CudaForge but leaves substantial performance on the table. Optimization-only feedback improves raw speed when it works, but correctness collapses because invalid kernels are never stabilized. The full workflow is the only variant that combines high correctness with high speedup.

### Comparative and Scaling Results

| Setting | Result |
| --- | --- |
| CudaForge on Levels 1 and 2 | `98.0%` correctness, `1.776x` performance |
| Agentic Baseline on Levels 1 and 2 | `95.0%` correctness, `1.490x` performance |
| CudaForge vs Kevin-32B on H200 (Levels 1 and 2 comparable tasks) | `98.0% / 1.662x` vs `82.0% / 1.10x` |
| Level 3 only on RTX 6000 | `96%` correctness, `1.283x` performance |
| Scaling rounds from 10 to 30 on subset | average speedup rises to `2.271x` |

These results support two different claims. First, a training-free hardware-guided agent can beat both a strong RL system and a stronger but more expensive prior agentic baseline. Second, the workflow has meaningful test-time scaling: more rounds continue to improve speed, though with diminishing returns after around 10 iterations.

### Cost, Metrics, and Generalization

| Item | Result |
| --- | --- |
| Average API cost per kernel | `$0.30` |
| Average wall-clock time per kernel on one RTX 6000 | `26.5` minutes |
| Prior agentic baseline reference | about `6 H100 hours` and `$5` per kernel |
| Full NCU metrics variant on subset | worse performance, about `40` minutes and `$1` per kernel |
| Cross-GPU subset results | `100%` correctness on RTX 6000, RTX 4090, A100, and RTX 3090 |
| Best reported cross-GPU average perf | `1.841x` on A100 |

The metric-selection result is an important part of the paper. Passing all NCU metrics to the Judge does not help; it hurts both quality and cost. The selected 24-metric subset gives the Judge enough hardware context to diagnose bottlenecks without drowning it in redundant signals. The cross-GPU results then show why the hardware-feedback design matters: because the Judge receives device-specific specs and profiler outputs, the workflow can retarget its reasoning to different GPUs without retraining.

## Limitation

The paper is convincing on practical engineering value, but the evidence has several limits.

| Limitation | Why It Matters |
| --- | --- |
| The main system is centered on NVIDIA tooling and Nsight Compute | The approach is not obviously portable to non-NVIDIA accelerators without redesigning the feedback interface |
| The strongest cost claims compare against a prior agentic baseline reported in a different environment | The cost gap is compelling but not perfectly apples-to-apples |
| The Kevin comparison is explicitly not fully fair or reproducible | The paper itself notes that Kevin-32B is not open-sourced, so the comparison has unavoidable uncertainty |
| Performance on Level 3 is positive but more modest than Level 2 | The workflow remains noticeably stronger on easier and mid-complexity kernels than on full architecture tasks |
| Manual inspection is still used for “fake kernels” in related systems | This suggests benchmark integrity remains a broader open problem beyond this workflow |

Another constraint is architectural. CudaForge’s Judge is effective because it has carefully designed inputs and a curated metric subset. That makes the workflow practical, but also means the method depends on prompt and metric engineering rather than on a more general learned representation of hardware bottlenecks.

---

*Reading date: 2026-04*
*Note status: Completed*
