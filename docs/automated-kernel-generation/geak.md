# GEAK: Introducing Triton Kernel AI Agent & Evaluation Benchmarks

<div class="paper-meta" markdown>

**Authors**: Jianghui Wang, Vinay Joshi, Saptarshi Majumder, Xu Chao, Bin Ding, Ziqiong Liu, Pratik Prabhanjan Brahma, Dong Li, Zicheng Liu, Emad Barsoum  
**Institution**: Advanced Micro Devices, Inc. (AMD)  
**Conference**: arXiv  
**Year**: 2025  
**Paper Link**: [arXiv:2507.23194](https://arxiv.org/abs/2507.23194)  
**GitHub**: [AMD-AIG-AIMA/GEAK-agent](https://github.com/AMD-AIG-AIMA/GEAK-agent)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Triton</span>
<span class="paper-tag">AMD GPUs</span>
<span class="paper-tag">Agentic Kernels</span>
</div>

## Background

GEAK addresses a specific gap in AI-assisted kernel generation: most public systems focus on CUDA or on generic correctness-oriented code generation, while AMD-targeted Triton kernels remain comparatively underexplored. The paper positions Triton as a useful target language because it balances performance with a higher-level programming interface, but still requires nontrivial hardware-specific knowledge to produce efficient kernels on devices such as MI300X and MI250.

The work contributes both an agentic kernel-generation framework and a benchmark suite. That dual contribution matters because raw correctness numbers are easy to overstate when test coverage is weak. GEAK therefore emphasizes not only model-side improvements, but also better evaluation methodology through benchmark revision and a new ROCm-focused benchmark.

**Key Takeaways**

- GEAK combines multiple agent roles with inference-time compute scaling to improve Triton kernel generation on AMD GPUs.
- The paper introduces two evaluation suites: TritonBench-revised and a new ROCm Triton benchmark built from real open-source AMD repositories.
- GEAK substantially outperforms direct frontier-model prompting and Reflexion-style baselines, reaching up to 63.33% correctness and up to 2.59x speedup on the revised benchmarks.

## Methodology

GEAK is structured as a modular agent pipeline for Triton kernel generation. The system takes a task description, generates candidate kernels, validates them, reflects on failures, and then applies optimization-oriented reasoning to improve latency and computational efficiency. The framework is designed around inference-time scaling rather than extra training, so improvements come from orchestrating stronger prompting, debugging, and optimization loops across frontier models.

A major part of the methodology is benchmark and evaluation design. The paper revises TritonBench-G to make kernels AMD-compliant, fixes missing calls and inconsistent tests, and adds tolerance-based tensor comparisons with stable random seeds. It also introduces a 30-kernel ROCm Triton benchmark sourced from public repositories such as ROCm/triton, ROCm/aiter, ROCm/aotriton, ROCm/vllm, ROCm/pytorch, ROCm/xformers, bitsandbytes, and TransformerEngine.

### System Overview

| Module | Input | Output | Role |
| --- | --- | --- | --- |
| Generator | Task query plus contextual prompt | Candidate Triton kernel | Produces initial code from minimal task descriptions |
| Evaluator | Candidate kernel and test harness | Functionality and performance results | First checks correctness, then profiles latency and memory behavior |
| Reflector | Failed kernel plus error trace | Debugging guidance or revised code direction | Analyzes execution failures and supports iterative correction |
| Optimizer | Functionally correct code plus historical performance | Optimization strategy and improved kernel candidates | Improves latency and efficiency using previous results |
| Parallel Scaling Layer | Multiple independent GEAK runs | Diverse kernel candidates | Explores broader search space through parallel generation with temperature sampling |

### Core Techniques

| Technique | Purpose | Effect |
| --- | --- | --- |
| One-shot Prompting | Retrieve a similar Triton example from non-overlapping data | Improves accuracy over zero-shot prompting |
| Knowledge Injection | Add low-level Triton and hardware optimization guidance to prompts | Gives a large lift in correctness and code quality |
| Reflexion-style Debugging | Feed error traces back into a self-correction loop | Repairs failing kernels through iterative analysis |
| LLM as Optimizer | Present prior code and performance history in ranked order | Guides the model toward better optimization directions |
| Debugging Trap Avoidance | Cap repeated attempts on the same failing path | Forces exploration of new strategies instead of repeated failure loops |
| Parallel Scaling | Run diverse GEAK instances independently | Improves chance of discovering correct and faster kernels |

### Benchmark Design

| Benchmark | Size | Purpose |
| --- | --- | --- |
| TritonBench-revised | 184 kernels | AMD-compatible revision of TritonBench-G with corrected tests and execution harness |
| ROCm Triton Benchmark | 30 kernels | Real-world Triton kernels collected from ROCm ecosystem repositories |

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Target Hardware | AMD Instinct MI300X and MI250 class GPUs |
| Code Target | Triton kernels for AMD GPUs |
| Baseline Models | Frontier LLMs such as GPT-4.1, Gemini 2.5 Pro, Claude 3.7 Sonnet |
| Primary Metrics | Call accuracy, execution accuracy, speedup |
| Evaluation Benchmarks | TritonBench-revised and ROCm Triton Benchmark |
| Scaling Strategy | Sequential Reflexion-style debugging plus parallel candidate generation |

### Headline Results

| Benchmark / Setting | Main Result |
| --- | --- |
| TritonBench-revised | Up to 54.89% execution accuracy |
| ROCm Triton Benchmark | Up to 63.33% correctness |
| TritonBench-revised speedup | Up to 2.59x average speedup over reference kernels |
| Direct prompting baseline | Usually below 15% correctness on the reported settings |
| GPT-4.1 on ROCm benchmark | Fails to generate any valid kernels in direct prompting baseline |

The main message is that GEAK clearly outperforms direct frontier-model prompting on both correctness and runtime efficiency. The improvement is especially meaningful on AMD-oriented Triton kernels, where direct prompting remains fragile and often fails to produce executable outputs at all.

### Module Ablation

| Knowledge Injection | 1-shot | Optimizer | Call Acc. (%) | Exec Acc. (%) | Speedup |
| --- | --- | --- | --- | --- | --- |
|  |  |  | 14.67 | 8.70 | 0.52 |
| ✓ |  |  | 52.72 | 20.11 | 0.86 |
| ✓ | ✓ |  | 54.35 | 27.17 | 0.99 |
| ✓ | ✓ | ✓ | 56.52 | 40.76 | 1.45 |

The ablation results show a clean division of labor between modules. Knowledge injection gives the first major jump in correctness, one-shot prompting continues improving accuracy, and the Optimizer contributes the largest improvement in speedup. The full combination is the only configuration that produces both strong correctness and clearly positive runtime gains.

### What the Results Suggest

| Observation | Interpretation |
| --- | --- |
| Direct prompting is weak even for strong frontier models | Kernel generation for Triton on AMD GPUs is still too brittle for naïve prompting |
| Knowledge injection has a large effect | Low-level domain knowledge remains essential |
| Optimizer has the strongest speedup contribution | Correctness alone is not enough; runtime feedback and optimization reasoning are necessary |
| ROCm benchmark matters | Better test coverage changes how trustworthy correctness claims really are |

## Limitation

GEAK improves both kernel generation and evaluation practice, but the paper also makes it clear that correctness reporting depends heavily on benchmark quality and test coverage. That disclaimer is one of the most useful parts of the work.

| Limitation | Why It Matters |
| --- | --- |
| Correctness depends strongly on test coverage | Weak benchmarks can overstate kernel validity and hide subtle numerical bugs |
| TritonBench-revised still inherits narrow coverage in places | Reported pass rates may look stronger than true general correctness |
| Speedups are moderate rather than dramatic | GEAK improves efficiency, but does not yet close the gap to expert kernels across the board |
| AMD focus narrows generality | The framework is specialized to Triton on AMD hardware rather than a universal kernel-generation solution |
| Inference-time scaling increases compute cost | Parallel generation and repeated reflection improve quality, but also raise inference expense |

---

*Reading date: 2026-04*
*Note status: Completed*
