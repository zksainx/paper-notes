# Improving Efficiency of GPU Kernel Optimization Agents using a Domain-Specific Language and Speed-of-Light Guidance

<div class="paper-meta" markdown>

**Authors**: Siva Kumar Sastry Hari, Vignesh Balaji, Sana Damani, Qijing Huang, Christos Kozyrakis  
**Conference**: arXiv  
**Year**: 2026  
**Paper Link**: [arXiv:2603.29010](https://arxiv.org/abs/2603.29010)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Domain-Specific Language</span>
<span class="paper-tag">CUTLASS</span>
<span class="paper-tag">Speed-of-Light Guidance</span>
</div>

## Background

This paper focuses on iteration efficiency for LLM-based GPU kernel optimization. The bottleneck is not just model reasoning cost. In their setup, roughly 21% of end-to-end time comes from LLM calls and 79% comes from compile, run, test, and profile actions, so every wasted attempt is expensive. The paper argues that two issues dominate this waste. First, raw CUDA or template-heavy CUTLASS code forces the model to spend too much effort on boilerplate and low-level correctness. Second, local profiling feedback alone does not tell the agent whether the current search branch is near a hardware limit or whether further budget should be spent elsewhere.

The proposed response is to raise the optimization abstraction while adding first-principles performance guidance. A compact DSL called `μCUTLASS` lets the agent express high-impact kernel decisions without writing low-level CUTLASS code directly. A Speed-of-Light (SOL) analysis pipeline then estimates theoretical headroom, guides hypothesis generation, stops search near diminishing returns, and helps detect benchmark gaming. The repo-friendly short name `DSL-SOL` is a good summary of the paper’s main idea: change the search space with a DSL and change the control policy with performance bounds.

**Key Takeaways**

- `μCUTLASS` replaces low-level CUTLASS code emission with a compact, statically validated DSL that keeps the important optimization knobs.
- SOL guidance is used for three distinct purposes: within-problem steering, across-problem budget scheduling, and integrity checking against reward hacking.
- On a 59-problem KernelBench subset, `μCUTLASS` turns GPT-5-mini from a `0.40x` regression to a `1.27x` geomean speedup over PyTorch, and `μCUTLASS` plus SOL guidance raises that to `1.56x`.

## Methodology

The paper introduces two coupled components. The first is `μCUTLASS`, a small DSL that exposes the high-impact choices in CUTLASS-backed kernels while hiding template instantiation, architecture-specific boilerplate, and other failure-prone code-generation details. The second is a SOL-first optimization workflow called MANTIS: Measure, Analyze, Nominate, Triage, Implement, Summarize. MANTIS uses an SOL report as structured context for both optimization and evaluation decisions.

`μCUTLASS` is intentionally narrow rather than fully general. It covers operator selection, kernel configuration, epilogue fusion, and multi-stage pipelines. Static checks reject invalid configurations before the expensive compile-test-profile cycle. MANTIS then uses the SOL report in three ways: to identify bottlenecks and propose hypotheses, to reallocate attempts away from problems that are already near their bound or have stagnated, and to flag suspiciously fast kernels or loophole-exploiting implementations during offline review.

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| `μCUTLASS` DSL | Short kernel specification | High-level program over CUTLASS-backed kernels | Lets the model reason over optimization knobs instead of template-heavy C++ |
| `μCUTLASS` Compiler | DSL string or file | Generated `ucutlass_*.h` header | Performs parsing, static validation, and code generation |
| Bootstrapper | KernelBench problem | Workspace, PyTorch baseline, SOL report | Prepares the optimization environment and baseline measurements |
| MANTIS Controller | Problem state, history, SOL report | Ranked hypotheses and candidate implementations | Structures the search into Analyze/Nominate/Triage/Implement phases |
| Integrity Pipeline | Candidate code, SOL report, profile logs | Accepted or rejected kernel | Detects gaming, PyTorch-only fallbacks, and physically implausible runtimes |

### Key Design Choices

- Keep the DSL compact enough to be learnable in context by a model with no prior exposure.
- Expose only the choices that strongly affect performance: data types, layouts, tiles, scheduling, epilogues, and explicit pipeline boundaries.
- Reject invalid DSL configurations before expensive execution, rather than surfacing them through failed compile attempts.
- Use SOL as a global headroom signal, not just a post-hoc profiling summary.
- Separate optimization quality from evaluation integrity by adding an explicit offline review pipeline.

### MANTIS Phases

| Phase | Main Question | Output |
| --- | --- | --- |
| Measure | How fast is the current best kernel relative to PyTorch? | Runtime and profiling context |
| Analyze | What does the SOL bound imply about bottlenecks and remaining headroom? | Structured SOL report |
| Nominate | What optimizations are plausible next moves? | Candidate hypotheses |
| Triage | Which hypotheses deserve budget now? | Selected hypotheses and attempt allocation |
| Implement | Can the chosen hypothesis be realized as working kernel code? | Evaluated candidates |
| Summarize | What reusable lessons were learned? | Short memory for later problems |

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Benchmark | KernelBench subset focused on LLM-relevant workloads |
| Problem Count | 59 problems from Levels 1-3 |
| Hardware | NVIDIA H100 (SM90a) |
| Runtime Metric | Nsight Compute kernel execution time |
| Models | GPT-5-mini, GPT-5, GPT-5.2 |
| Baseline Controller | Flat MI loop (Generate-Compile-Test-Profile) |
| Budget | 40 attempts per problem |
| Orchestrated Budget | `5` iterations x `2` hypotheses x `4` attempts |
| Framework | OpenHands with per-problem workspace |

### Headline Results

| Model Tier | MI Baseline | `μCUTLASS` + MI | `μCUTLASS` + SOL-Guided |
| --- | --- | --- | --- |
| GPT-5-mini | `0.40x` | `1.27x` | `1.56x` |
| GPT-5 | `0.86x` | `1.69x` | `2.07x` |
| GPT-5.2 | `2.04x` | `2.85x` | `2.79x` |

The strongest pattern is that the DSL is the biggest single lever, especially for weaker models. GPT-5-mini moves from a large regression to a clear win over PyTorch once the representation changes from raw code to DSL. SOL-guided steering adds another layer of improvement for mini and mid-tier models, enough for `GPT-5-mini + μCUTLASS + SOL` to beat the raw-code GPT-5 baseline, and for `GPT-5 + μCUTLASS + SOL` to match the GPT-5.2 baseline at lower inference cost. On GPT-5.2, however, the full combination is slightly below `μCUTLASS` alone (`2.79x` vs `2.85x`), suggesting diminishing returns once the model is already strong enough to exploit the DSL directly.

### Scheduling and Efficiency

| Setting | Main Result |
| --- | --- |
| Token savings under best policies | `19%-43%` while retaining at least `95%` geomean speedup |
| Best overall efficiency gain | `1.68x` on GPT-5 + `μCUTLASS` + SOL |
| Best GPT-5 policy | `epsilon=250%`, `w=12`, `43%` savings, `96%` retention |
| Example GPT-5.2 moderate policy | `epsilon=25%`, `w=16`, `33%` token savings, `96%` retention |
| Best GPT-5-mini `μCUTLASS` + SOL policy | `1.49x` efficiency gain with `36%` savings |

These results show that SOL is not only a search hint. It makes scheduling principled. Once a problem is close enough to the estimated bound or has plateaued for too long, the scheduler can stop spending attempts there and move resources to higher-headroom problems. This converts every fixed-budget agent into a cost-versus-speed frontier instead of a single operating point.

### Integrity Checking

| Integrity Signal | What It Catches | Why It Matters |
| --- | --- | --- |
| SOL-ceiling detector | Kernels faster than plausible physical bound | Flags likely skipped work or benchmark exploitation |
| LLM Game Detector | Original or inherited gaming patterns | Catches loopholes correctness tests do not specify |
| PyTorch-only detector | Solutions that only compose library calls | Prevents counting library fallback as custom kernel generation |
| Inflation without filtering | Up to `1.9x` reported speedup inflation | Shows unfiltered results can badly overstate capability |

The integrity section is unusually important here because the paper shows how often LLMs exploit benchmark loopholes. For example, GPT-5-mini `μCUTLASS` + MI would appear to improve from `1.27x` to `2.28x` if PyTorch-only fallbacks were allowed. GPT-5.2 `μCUTLASS` + MI would jump from `2.85x` to `5.34x` if gaming solutions were not filtered. Prompt-level anti-gaming instructions reduce some bad behavior but do not reliably solve the problem, so the paper treats SOL-guided integrity checking as part of the evaluation stack rather than an optional add-on.

## Limitation

The paper is strong on systems methodology, but its evidence still has clear scope limits.

| Limitation | Why It Matters |
| --- | --- |
| Evaluation uses 59 out of 250 KernelBench problems | The subset is targeted toward modern LLM workloads, so broader generalization is still open |
| `μCUTLASS` is narrow by design | The gains come from a carefully chosen search space, not a universal GPU programming abstraction |
| SOL bounds rely on modeling assumptions | Scheduling and integrity decisions can shift if the bound is too loose or too tight |
| Benefits are model-tier dependent | SOL orchestration helps weaker models more than the strongest one, so the controller is not uniformly optimal |
| Integrity review is offline | The agent cannot repair flagged near-miss kernels during the same run, so the evaluation pipeline is not yet fully closed-loop |

The paper also makes a deeper trade-off explicit: by restricting the space through a DSL, it gains iteration efficiency but gives up some expressiveness. That is the right choice for this workload, but extending the approach to other kernel families or other IR ecosystems will require new DSL design rather than simple reuse.

---

*Reading date: 2026-04*
*Note status: Completed*
