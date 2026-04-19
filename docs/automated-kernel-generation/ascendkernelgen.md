# AscendKernelGen: A Systematic Study of LLM-Based Kernel Generation for Neural Processing Units

<div class="paper-meta" markdown>

**Authors**: Xinzi Cao, Jianyang Zhai, Pengfei Li, Zhiheng Hu, Cen Yan, Bingxu Mu, Guanghuan Fang, Bin She, Jiayu Li, Yihan Su, Dongyang Tao, Xiansong Huang, Fan Xu, Feidiao Yang, Yao Lu, Chang-Dong Wang, Yutong Lu, Weicheng Xue, Bin Zhou, Yonghong Tian  
**Conference**: arXiv  
**Year**: 2026  
**Paper Link**: [arXiv:2601.07160](https://arxiv.org/abs/2601.07160)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">NPU Kernels</span>
<span class="paper-tag">AscendC</span>
<span class="paper-tag">Domain Adaptation</span>
</div>

## Background

AscendKernelGen addresses a domain where general LLM-based code generation is especially brittle: low-level kernel development for neural processing units. The paper focuses on Huawei Ascend NPUs, where kernel programming uses AscendC and requires precise reasoning about asynchronous pipelines, hierarchical memory movement, explicit synchronization, and hardware-specific API contracts. These constraints make NPU kernel generation significantly harder than ordinary code synthesis and even harder than many existing GPU-kernel benchmarks.

The paper argues that current general-purpose models fail not because of syntax alone, but because they lack the structured reasoning needed for NPU kernel programming. On the authors' zero-shot evaluation, general LLMs are effectively unusable for non-trivial Ascend kernels, with near-zero success on complex tasks. AscendKernelGen is proposed as a full generation-evaluation framework to bridge that gap through domain-specific data, domain-adaptive post-training, and hardware-grounded benchmarking.

**Key Takeaways**

- AscendKernelGen combines dataset construction, model adaptation, and evaluation into a unified framework for NPU kernel generation.
- The system introduces `Ascend-CoT` for reasoning-rich supervision and `NPUKernelBench` for compilation, correctness, and performance evaluation.
- On complex Level-2 kernels, the reported compilation rate improves from 0% to 95.5% (Pass@10), with functional correctness reaching 64.3% where general models previously failed completely.

## Methodology

AscendKernelGen is organized around three tightly coupled pieces: a reasoning-oriented dataset, a domain-adapted generation model, and a dedicated evaluation subsystem. The goal is not just to make the model imitate AscendC syntax, but to teach the reasoning patterns needed for correct low-level kernel construction, such as pipeline construction, synchronization placement, and boundary-sensitive index arithmetic.

The generation side is built from `Ascend-CoT` and `KernelGen-LM`. Ascend-CoT combines documentation-derived reasoning, code-centric chain-of-thought traces from real Ascend kernels, and general reasoning data. KernelGen-LM is then trained through supervised fine-tuning followed by reinforcement learning with execution feedback, so the model first learns structural competence and then refines execution-level behavior using hardware-grounded preference signals.

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Ascend-CoT | Documentation, real AscendC kernels, reasoning traces | Domain-specific supervision data | Teaches hardware-aware reasoning rather than only syntax imitation |
| KernelGen-LM | Base LLM plus domain-adaptive training | NPU-specialized code model | Generates Ascend kernels with stronger compilation and execution behavior |
| NPUKernelBench | Generated kernels and task definitions | Compilation, correctness, and performance results | Provides staged evaluation on real hardware |
| Logging and Feedback Loop | Compiler logs, verifier output, runtime measurements | Fine-grained diagnostics | Supports both error analysis and execution-guided RL |

### Ascend-CoT Data Construction

| Data Source | What It Contributes |
| --- | --- |
| Documentation-based CoT | API semantics, programming model, hardware abstractions, best practices |
| Code-centric CoT | Real host/kernel interaction patterns, synchronization logic, tiling and memory reasoning |
| General CoT data | Preserves broader reasoning capability during specialization |

### Key Design Choices

- NPU kernels are treated as structured programs that jointly encode data partitioning, pipeline stages, synchronization, and layout reasoning.
- Supervised fine-tuning teaches the model to produce valid AscendC structure and API usage before RL tries to improve execution outcomes.
- RL uses execution-based preference signals rather than only compile success, which is especially important for subtle synchronization and memory-ordering errors.
- NPUKernelBench uses a dual-path evaluation design so both kernel-only optimization and host-device integration can be studied rigorously.

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Target Platform | Huawei Ascend NPU stack using AscendC |
| Evaluation Benchmark | NPUKernelBench |
| Benchmark Size | 158 kernels across three complexity levels |
| Base Model | Qwen3-32B in the main reported setting |
| Training Stages | Base model, SFT, SFT + RL |
| Primary Metrics | Compilation Rate (CR), Execution Rate (ER), Pass@k, Speedup |

### Headline Results

| Setting | Key Result |
| --- | --- |
| Zero-shot general models on complex kernels | Near-zero success |
| Level-2 complex kernels after training | 95.5% compilation success at Pass@10 |
| Level-2 functional correctness | 64.3% |
| Base model Pass@1 on representative kernels | 7.92% |
| SFT Pass@1 on representative kernels | 26.26% |
| SFT + RL Pass@1 on representative kernels | 33.46% |

The paper's core message is that domain adaptation matters much more than simply scaling a general-purpose coder. Without hardware-specific training, the model mostly fails. Once Ascend-specific reasoning is injected through SFT and then sharpened through execution-guided RL, the system becomes capable of compiling and correctly running many kernels that were previously out of reach.

### Performance Speedup

| Model Stage | Level 1 Speedup | Level 2 Speedup | Level 3 Speedup |
| --- | --- | --- | --- |
| Base Qwen3-32B | 0.60x | 0.00x | 0.00x |
| Qwen3-32B + SFT | 0.56x | 1.50x | 0.00x |
| Qwen3-32B + SFT + RL | 0.61x | 1.86x | 0.00x |

The strongest optimization gains appear on Level-2 kernels. That is a meaningful result because Level-2 tasks are complex enough to expose real hardware-aware opportunities, but still structured enough for the model to learn from data and execution feedback. Level-1 kernels have less optimization headroom, while Level-3 kernels remain extremely hard because of larger control-flow and dependency complexity.

### Fine-Tuning and RL Ablations

| Ablation | Main Finding |
| --- | --- |
| Full tuning vs. LoRA | Full tuning improves mean compile rate from 40.29% to 55.32% and mean exec rate from 13.55% to 22.13% |
| RL negative data choice | Using compile-pass but execution-fail samples as negatives is better than using pure compile-fail negatives |
| RL schedule | Cosine decay performs better than a constant learning-rate schedule |
| Model scale | 32B is the only tested scale with meaningful compile success on Level-3 kernels |

These ablations support the claim that NPU kernel generation is not a lightweight style-transfer problem. The task requires deep parameter updates and execution-grounded supervision because the model must learn strict API semantics, memory rules, and synchronization logic rather than only local token patterns.

## Limitation

AscendKernelGen demonstrates strong progress on Ascend NPU kernels, but the paper also makes clear that this is still a difficult setting where execution success on the hardest kernels remains limited.

| Limitation | Why It Matters |
| --- | --- |
| Institution metadata is not recoverable from the converted source | The converted Markdown preserves authors but does not expose affiliations clearly enough for confident recovery |
| Level-3 kernels remain very challenging | Even after SFT + RL, execution success on the hardest kernels is still limited |
| Performance gains are concentrated on Level-2 tasks | The strongest speedups do not generalize uniformly across all difficulty levels |
| Hardware and language specificity are very high | Results target AscendC and Ascend NPUs rather than a general accelerator family |
| API signature errors dominate failures | The largest remaining failure category is still hardware-specific semantic misuse rather than syntax |

---

*Reading date: 2026-04*
*Note status: Completed*
