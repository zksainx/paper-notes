# CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation

<div class="paper-meta" markdown>

**Authors**: Weinan Dai, Hanlin Wu, Qiying Yu, Huan-ang Gao, Jiahao Li, Chengquan Jiang, Weiqiang Lou, Yufan Song, Hongli Yu, Jiaze Chen, Wei-Ying Ma, Ya-Qin Zhang, Jingjing Liu, Mingxuan Wang, Xin Liu, Hao Zhou  
**Institution**: ByteDance Seed; Institute for AI Industry Research (AIR), Tsinghua University  
**Conference**: arXiv  
**Year**: 2026  
**Paper Link**: [arXiv:2602.24286](https://arxiv.org/abs/2602.24286)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">CUDA Kernel Optimization</span>
<span class="paper-tag">Agentic Reinforcement Learning</span>
<span class="paper-tag">KernelBench</span>
</div>

## Background

CUDA kernel optimization remains a bottleneck for production deep learning systems because performance depends on hardware-aware implementation details that are hard to recover from generic code reasoning alone. Existing LLM-based approaches either run training-free refinement loops or fine-tune models with fixed feedback templates, but both lines often plateau below strong compiler baselines such as `torch.compile`.

CUDA Agent targets this gap by training an LLM as a long-horizon coding agent in a CUDA-specific tool environment. The central claim is that optimization ability should be learned end-to-end with execution feedback, rather than approximated by one-shot generation or hand-crafted search scripts.

**Key Takeaways**

- CUDA Agent combines data scaling, agent-environment design, and RL stability tricks into one training system instead of treating them as separate patches.
- The system explicitly optimizes both correctness and speed with anti-hacking guardrails in the reward pipeline.
- On KernelBench, the model substantially exceeds `torch.compile` and strong proprietary coding models, especially on hard fused operators.

## Methodology

CUDA Agent is built as a three-part pipeline: synthesized training tasks, a skill-integrated agent loop, and stable agentic RL. The training dataset (`CUDA-Agent-Ops-6K`) is designed to expose fusion-heavy operator compositions that require nontrivial kernel-level decisions.

The interaction loop follows a ReAct-style coding workflow over tools, with explicit profiling, correctness checks, and iterative code revision. RL then optimizes long multi-turn trajectories, with warm-up stages for both actor and critic to prevent early collapse.

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Data Synthesis Pipeline | Seed operators from `torch` and `transformers` | `CUDA-Agent-Ops-6K` (6,000 filtered tasks) | Builds diverse operator-level RL tasks through composition + filtering |
| Skill-Integrated Agent Loop | Task spec, coding tools, profiling scripts | Multi-turn optimization trajectories | Enables analyze-implement-profile-refine behavior under execution feedback |
| Robust Reward Scheduler | Correctness results + eager/compile runtime | Discrete reward in `{-1, 1, 2, 3}` | Stabilizes optimization signal across tasks with different difficulty |
| Anti-Hacking Evaluation Guardrails | Candidate kernels and eval scripts | Trustworthy reward signal | Prevents reward hacking and fallback-based false speedups |
| Multi-Stage RL Training | Single-turn warm-up traces + agent trajectories | Trained CUDA Agent policy | Uses RFT actor init + value pretraining + PPO for stable long-horizon RL |

### Key Design Choices

| Design Choice | Concrete Implementation | Why It Matters |
| --- | --- | --- |
| Task-space scaling | Crawl seed ops, synthesize fused ops, filter by executability/determinism/nontrivial runtime | Creates enough high-quality tasks for RL without relying on KernelBench training data |
| Agent skills abstraction | CUDA workflow encoded as skill instructions + tool use | Teaches iterative optimization behavior, not just final-code emission |
| Milestone reward | Reward tiers based on correctness and >=5% wins vs eager/compile | Reduces outlier sensitivity compared with raw speedup ratio reward |
| Training stability | Single-turn RL warm-up, RFT for actor, value pretraining for critic | Avoids policy collapse in long-context multi-turn optimization |
| Secure evaluation | Script protection, fallback restrictions, multi-input checks, synchronized profiling | Improves validity of measured speedup and prevents shortcut exploitation |

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Benchmark | KernelBench Level 1-3 (250 tasks total) |
| Baselines | `torch` eager, `torch.compile`, Claude Opus 4.5, Gemini 3 Pro, GLM 4.6, Kimi K2 |
| Main metrics | Pass Rate, Faster Rate, Geomean Speed-up |
| Base model | Seed1.6 (23B active / 230B total MoE) |
| RL training scale | 150 steps, batch size 1024 |
| Context / turns | 32,768 (single-turn RL), 131,072 (agentic RL), up to 150 turns train / 200 eval |
| Sandbox infra | CPU-GPU decoupled execution with 128 NVIDIA H20 GPUs |

### Headline Results

| Setting | Pass Rate | Faster Rate vs Compile | Geomean Speed-up vs Compile |
| --- | --- | --- | --- |
| Overall CUDA Agent | 98.8% | 96.8% | 2.11x |
| Level 1 | 100.0% | 97.0% | 1.87x |
| Level 2 | 100.0% | 100.0% | 2.80x |
| Level 3 | 94.0% | 90.0% | 1.52x |

Compared with proprietary baselines, CUDA Agent reports materially higher optimization success at the hard end of the benchmark. In Level 3, Faster Rate vs `torch.compile` is 90.0% for CUDA Agent versus 50.0% (Claude Opus 4.5) and 52.0% (Gemini 3 Pro), matching the paper's statement of about 40-point advantage on the hardest split.

### Ablation Results

| Variant | Pass Rate | Faster Rate vs Compile | Geomean Speed-up vs Compile |
| --- | --- | --- | --- |
| w/o Agent Loop | 77.1% | 14.1% | 0.69x |
| w/o Robust Reward | 96.8% | 60.4% | 1.25x |
| w/o RFT | 95.6% | 49.8% | 1.05x |
| w/o Value Pretraining | 98.6% | 50.9% | 1.00x |
| Full CUDA Agent | 98.8% | 96.8% | 2.11x |

Ablations show that correctness alone is not enough for strong optimization: removing reward design or multi-stage stabilization preserves high pass rates but sharply reduces speedup quality. Removing the agent loop causes both correctness and performance regression, indicating that multi-turn execution feedback is central to the gains.

## Limitation

| Limitation | Why It Matters |
| --- | --- |
| No comparison with stronger compiler stacks like TVM | Leaves open whether gains hold against more advanced autotuning compilers |
| High infrastructure cost (large isolated GPU pool) | Makes direct reproduction difficult for smaller labs |
| Training system complexity | Data synthesis, secure sandboxing, and multi-stage RL increase engineering burden |
| Institution metadata ambiguity in converted source | The extracted text did not preserve explicit affiliation block, which weakens metadata completeness |

The paper itself explicitly highlights the first two points; the latter two are practical implications from the reported pipeline design and artifact format.

---

*Reading date: 2026-04*
*Note status: Completed*
