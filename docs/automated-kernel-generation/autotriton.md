# AutoTriton: Automatic Triton Programming with Reinforcement Learning in LLMs

<div class="paper-meta" markdown>

**Authors**: Shangzhan Li, Zefan Wang, Ye He, Yuxuan Li, Qi Shi, Jianling Li, Yonggang Hu, Wanxiang Che, Xu Han, Zhiyuan Liu, Maosong Sun  
**Institution**: Tsinghua University, Harbin Institute of Technology, Tianjin University, OpenBMB  
**Conference**: arXiv  
**Year**: 2025  
**Paper Link**: [arXiv:2507.05687](https://arxiv.org/abs/2507.05687)  
**GitHub**: [AI9Stars/AutoTriton](https://github.com/AI9Stars/AutoTriton)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Triton</span>
<span class="paper-tag">Reinforcement Learning</span>
<span class="paper-tag">Kernel Generation</span>
</div>

## Background

AutoTriton studies automatic Triton programming: given a PyTorch implementation or operator specification, generate executable Triton code that is both semantically correct and reasonably fast. Triton removes much of CUDA's boilerplate, but real kernel development still depends on tiling, memory layout, launch structure, and syntax details that are hard for general-purpose LLMs to internalize from broad code data alone.

The paper argues that existing approaches hit two ceilings. Pure prompting or agentic workflows remain limited by the base model's Triton knowledge, while pure supervised fine-tuning mostly imitates existing solutions and leaves little room for exploration. AutoTriton addresses this by combining a Triton-specific data construction pipeline with RL from verifiable execution feedback.

**Key Takeaways**

- AutoTriton is framed as the first RL-trained model dedicated specifically to Triton programming.
- The training recipe is sequential: Triton-specific SFT for basic competence, then GRPO-based RL for correctness-oriented exploration.
- An 8B model reaches execution accuracy comparable to or better than much larger frontier models on several TritonBench and KernelBench channels.

## Methodology

AutoTriton uses a two-stage training pipeline built on top of `Seed-Coder-8B-Reasoning`. The target task is a mapping from kernel specification to Triton implementation. The core idea is to first teach Triton programming patterns explicitly, then use execution feedback to push the model beyond imitation learning.

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Data Collection Pipeline | PyTorch kernels from GitHub, HuggingFace, and synthesized operators | Verified PyTorch kernel pool | Builds executable source tasks with test cases |
| Distillation Path | PyTorch kernel + Triton knowledge prompt | Triton code + CoT explanation | Produces instruction-following examples through strong reasoning models |
| Compilation Refinement Path | `torch.compile` artifacts + LLM cleanup | Readable Triton kernels + CoT | Converts compiler-produced Triton into higher-quality supervision |
| SFT Stage | `<instruction, Triton code with CoT>` pairs | Triton-specialized base model | Teaches syntax, programming idioms, and reasoning patterns |
| RL Stage | `<instruction, PyTorch code>` pairs + test execution | RL-updated policy | Optimizes for valid Triton generation through verifiable rewards |

### Data Pipeline and SFT

The SFT dataset is built from two complementary sources. The first asks a strong LLM to translate verified PyTorch kernels into Triton with step-by-step reasoning. The second starts from `torch.compile` generated Triton, then uses an LLM to clean the code, generate instructions, and add explanatory CoT. Both paths are filtered by functional tests before entering training. The resulting dataset contains 14,102 samples and focuses on explicit Triton-specific reasoning rather than generic code completion.

### RL Objective and Reward Design

RL uses GRPO with a binary reward:

- `1` if the output is valid Triton and passes test cases
- `0` otherwise

The reward combines two checks:

- `is_Triton`: a rule-based syntactic check that discourages falling back to PyTorch or other non-Triton code
- `test_passed`: execution-based validation against the reference PyTorch implementation

This split is important because execution-only rewards are easy to game. The paper shows failure modes where the model emits a trivial Triton stub or wraps the real computation in plain Torch while still satisfying tests. The rule-based reward reduces that kind of reward hacking and stabilizes RL.

### Key Design Choices

- Use a Triton-specific dataset instead of relying on generic coder pretraining.
- Keep RL supervision lightweight and verifiable by using binary correctness rewards.
- Mix OOD RL data with a small amount of in-distribution SFT data to smooth policy transition.
- Optimize correctness first; performance is evaluated afterward rather than used directly in training.

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Backbone | Seed-Coder-8B-Reasoning |
| SFT Data | 14,102 samples |
| RL Data | 6,302 samples |
| SFT Hardware | 8 x A800 GPUs for about 16 hours |
| RL Hardware | 16 x A800 GPUs for about 32 hours |
| Benchmarks | TritonBench-G, TritonBench-T, KernelBench Level 1/2/3 |
| Metrics | Call / Compilation accuracy, Execution accuracy, `fast_1`, `fast_2` |

### Headline Results

| Benchmark Channel | AutoTriton Result | Comparison Signal |
| --- | --- | --- |
| TritonBench-G | 15.76% Exec, `fast_1` 7.61, `fast_2` 2.17 | Best execution accuracy among reported models |
| TritonBench-T | 40.36% Call, 39.16% Exec | Best call and execution accuracy |
| KernelBench Level 1 | 83% Comp, 36% Exec | Best execution accuracy; better than DeepSeek-R1-0528 at 35% |
| KernelBench Level 2 | 97% Comp, 45% Exec | Best execution accuracy; competitive speed |
| KernelBench Level 3 | 82% Comp, 20% Exec | Second-tier correctness, but strongest `fast_2` at 4.0 |

The main result is not raw speed dominance on every channel. Instead, AutoTriton consistently raises correctness on Triton generation tasks and often matches or exceeds much larger models despite having only 8B parameters. The gains are especially clear on TritonBench-T and KernelBench Levels 1-2, where specialized training matters more than general coding scale alone.

### Ablations and Analysis

| Ablation / Comparison | Main Finding |
| --- | --- |
| AutoTriton vs. SFT-only | RL improves execution accuracy on all three KernelBench levels and both TritonBench channels |
| AutoTriton vs. backbone | SFT already provides a large jump over Seed-Coder-8B-Reasoning |
| With vs. without rule-based reward | Invalid generations without `@triton.jit` drop from 18 to 5 on TritonBench-T and from 25 to 6 on KernelBench Level 1 |
| Triton vs. CUDA comparison | AutoTriton beats AI CUDA Engineer on Triton-formulated KernelBench, but still trails strong CUDA-oriented models such as Kevin-32B |

These results support a narrow but credible conclusion: RL helps once the model has already learned Triton syntax and programming structure through SFT, and reward design matters because correctness-only signals are exploitable.

## Limitation

The paper is explicit that current training is correctness-guided rather than performance-guided. That makes the model better at producing valid Triton, but it does not directly optimize runtime quality.

| Limitation | Why It Matters |
| --- | --- |
| No performance reward in RL | Correct kernels may still be slow or miss key tiling and memory optimizations |
| TritonBench-G remains hard for all models | Real GitHub kernels still expose a large gap between benchmark progress and robust real-world generation |
| Rule-based reward is still hackable | The model can satisfy surface syntax requirements with low-quality or partially fake Triton code |
| Cross-language gap persists | AutoTriton improves Triton generation, but CUDA-specialized systems still lead on some KernelBench comparisons |
| Heavy training cost | The pipeline requires multi-node A800 training rather than a lightweight post-training recipe |

---

*Reading date: 2026-04*
*Note status: Completed*
