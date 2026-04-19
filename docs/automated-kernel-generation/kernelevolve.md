# KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

<div class="paper-meta" markdown>

**Authors**: KernelEvolve Team  
**Institution**: Meta Platforms  
**Conference**: arXiv  
**Year**: 2025  
**Paper Link**: [arXiv:2512.23236](https://arxiv.org/abs/2512.23236)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Heterogeneous Hardware</span>
<span class="paper-tag">Agentic Search</span>
<span class="paper-tag">MTIA</span>
</div>

## Background

KernelEvolve is motivated by a production-scale kernel coverage problem rather than a single benchmark challenge. Meta’s ranking and recommendation systems run across a heterogeneous fleet that includes NVIDIA GPUs, AMD GPUs, and Meta’s custom MTIA accelerators. In that setting, kernel availability and kernel efficiency are both first-order constraints: missing kernels force fallback architectures, while slow kernels directly affect latency and infrastructure cost.

The paper frames this as a three-dimensional scalability crisis involving hardware diversity, model diversity, and operator diversity. Manual kernel development does not scale when each hardware platform requires many operator variants and each new generation changes the optimization playbook. KernelEvolve is presented as an agentic search system designed to close that hardware-software gap at production scale.

**Key Takeaways**

- KernelEvolve treats kernel generation as graph-based search over candidate implementations rather than isolated code generation.
- A universal operator plus retrieval-augmented context replaces rigid multi-operator prompting templates.
- The system is deployed against heterogeneous production workloads and reports 100% correctness across 480 operator-platform configurations, with 1.25x to 17x speedups on production workloads.

## Methodology

KernelEvolve formalizes search as a graph-based process over kernel implementation states. Each node is scored by a fitness function defined as speedup relative to a PyTorch compiled baseline, with correctness treated as a hard constraint. A selection policy chooses which nodes to expand, a universal operator generates new candidates from runtime context, and a termination rule stops the search once budget or progress limits are reached.

A central design claim is that prior multi-operator frameworks are bottlenecked by static prompt templates. Instead of maintaining separate hard-coded operators such as Draft, Debug, and Improve, KernelEvolve uses a single universal operator whose behavior is conditioned by retrieval-augmented runtime context. This lets the system respond to the actual bottleneck observed during execution rather than forcing the model into a pre-labeled operator role.

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Graph-Based Search | Kernel candidates and fitness scores | Search frontier | Organizes optimization as iterative expansion over candidate states |
| Selection Policy | Current search graph | Nodes selected for expansion | Supports greedy, MCTS-like, or evolutionary search strategies |
| Universal Operator | Program state plus runtime context | New kernel candidates | Replaces static operator templates with context-adaptive generation |
| Evaluation Framework | Candidate kernel execution | Correctness and speedup scores | Validates generated kernels against reference implementations |
| Retrieval-Augmented Context | Profiling traces, error diagnostics, hardware constraints | Context packets for the LLM | Supplies only the relevant knowledge needed for the current bottleneck |

### Agentic Retrieval and Self-Managed Context

| Sub-Agent | Main Function |
| --- | --- |
| Context Memory Sub-Agent | Interprets runtime artifacts such as profiling results, correctness failures, and diagnostics |
| Deep Search Sub-Agent | Retrieves targeted knowledge from a persistent hardware and optimization knowledge base |
| MTIA Knowledge Injection | Adds architecture-specific knowledge for proprietary accelerators absent from public model pretraining |

### Key Design Choices

- The universal operator is context-driven rather than tied to fixed operator categories.
- Retrieval is hierarchical: platform constraints, optimization guidance, and hardware-specific documents are loaded on demand.
- Profiling is multi-granular, including system-level, kernel-level, and intra-kernel views.
- KernelEvolve explicitly targets proprietary accelerators such as MTIA, where public training corpora provide little useful prior knowledge.

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Deployment Context | Meta recommendation and ranking workloads |
| Hardware Targets | NVIDIA GPUs, AMD GPUs, Meta MTIA |
| Workload Types | OSS operators, monetization case studies, data preprocessing kernels, sequence learning kernels |
| Objective | Speedup over PyTorch compiled or eager baselines with correctness as a hard constraint |
| Evaluation Infrastructure | Unified profiling, generated evaluation code, debugging in JIT flow, FaaS-based kernel evaluation |

### Headline Results

| Result Category | Reported Outcome |
| --- | --- |
| Overall correctness | 100% across 480 operator-platform configurations |
| Production workload speedups | 1.25x to 17x |
| Development time reduction | From weeks to hours |
| Production value proposition | Enables missing kernels on MTIA and accelerates existing supported kernels |

The headline contribution is less about a single benchmark win and more about breadth. KernelEvolve is presented as a production-oriented system that both unlocks unsupported operators on new accelerators and improves performance on already supported paths. That is a stronger systems claim than the usual “better on one benchmark” result.

### Representative Production Results

| Case | Hardware | Reported Behavior |
| --- | --- | --- |
| Convolutional and transformer-style monetization kernels | Heterogeneous production accelerators | 1.25x to 17x gains depending on operator and platform |
| MTIA MapIdTransform | MTIA v2i / v3 | Enables unsupported operator coverage and delivers up to about 4x speedup on some shapes |
| MTIA MergeBucketizedDenseTransform | MTIA v2i / v3 | 2x to 9x class speedups depending on configuration |
| Batch Event Truncate | Production sequence-learning workloads | Batched Triton kernel generated automatically from a non-batched PyTorch baseline |

### MTIA Kernel Coverage as a Systems Result

| Hardware | Operator Issue | KernelEvolve Value |
| --- | --- | --- |
| MTIA v2i | Missing ATen operators block deployment | Generated kernels provide the missing implementations required for on-device execution |
| MTIA v3 | More operator coverage but still performance headroom | Generated kernels improve latency through fusion and MTIA-specific tuning |

This is one of the paper’s most important points. On emerging hardware, automated kernel generation is not only an optimization tool; it is a deployment enabler. Missing kernels can force CPU fallback or even block rollout entirely, so synthesis quality directly affects system architecture.

## Limitation

KernelEvolve demonstrates scale and breadth, but the paper also makes clear that the system is complex and deeply tied to Meta’s production environment.

| Limitation | Why It Matters |
| --- | --- |
| Production environment dependence | Many evaluation cases are tied to Meta-specific workloads, infrastructure, and accelerators |
| Abstract is malformed in the converted source | The arXiv conversion replaces the abstract with placeholder text, so downstream summarization must rely on the body |
| Broad scope reduces per-task detail | The paper covers many operators and platforms, but individual kernel analyses are necessarily selective |
| Heavy system integration required | Retrieval, profiling, validation, FaaS execution, and hardware-specific injection make the framework nontrivial to reproduce |
| Cross-platform success still requires curated knowledge | The universal operator helps, but strong results still depend on rich hardware-specific guidance modules |

---

*Reading date: 2026-04*
*Note status: Completed*
