# Kevin: Multi-Turn RL for Generating CUDA Kernels

<div class="paper-meta" markdown>

**Authors**: Carlo Baronio, Pietro Marsella, Ben Pan, Simon Guo, Silas Alberti  
**Institution**: Stanford University  
**Conference**: arXiv  
**Year**: 2025  
**Paper Link**: [arXiv:2507.11948](https://arxiv.org/abs/2507.11948)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Multi-Turn RL</span>
<span class="paper-tag">CUDA Kernels</span>
<span class="paper-tag">KernelBench</span>
</div>

## Background

Kevin studies CUDA kernel generation as an inherently iterative optimization task rather than a one-shot code synthesis problem. Real performance engineers write a kernel, execute it, inspect failures or timing results, and then refine it over multiple rounds. Standard RL-for-code recipes usually train on single-turn outcomes, which makes them a poor fit for this workflow because early but promising intermediate kernels often receive little or no useful credit.

The paper proposes a multi-turn RL recipe that directly mirrors this refinement process. Instead of assigning reward only to a final answer, Kevin trains on every refinement turn and propagates value backward across the trajectory. The target setting is KernelBench-style CUDA generation, where correctness and runtime can both be measured automatically.

**Key Takeaways**

- Kevin is presented as the first model trained with multi-turn RL for CUDA kernel generation.
- Multi-turn RL improves both correctness and performance over the QwQ-32B base model and over a single-turn RL baseline.
- Under a fixed inference budget, sequential refinement is more effective than spending all compute on parallel one-shot sampling.

## Methodology

Kevin starts from `QwQ-32B` and trains with GRPO in an environment where the model repeatedly proposes CUDA kernels, receives execution feedback, and refines its solution. The central design goal is to make RL work on long-horizon engineering trajectories without losing sample efficiency or exploding the context window.

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Kernel Environment | PyTorch reference implementation | Compiled / executed / profiled CUDA candidate | Supplies correctness and speed feedback for each turn |
| Single-Turn RL Baseline | One kernel attempt per sample | GRPO-trained one-shot optimizer | Baseline that optimizes only first-pass generation |
| Multi-Turn Trajectory Builder | Prior kernels, summarized CoTs, evaluation feedback | New refinement prompt | Lets the model condition on previous attempts and outcomes |
| Turn-Wise Training | Each refinement turn in a trajectory | Individual RL training sample | Increases sample efficiency versus training only on final outcome |
| Discounted Reward Aggregation | Current and future kernel scores | Per-turn reward | Credits early suboptimal steps if they enable later strong kernels |

### Context Management and Turn-Wise Training

The main practical issue is trajectory length. Reasoning models generate long CoTs, and naively appending all previous reasoning quickly exceeds the context window. Kevin handles this by discarding earlier full CoTs and keeping only a short summary of each turn's reasoning changes, along with the generated kernels and their evaluation results.

Instead of treating an entire multi-turn trajectory as one RL sample, the paper splits it into per-turn samples:

- each turn gets its own training example
- the prompt contains the summarized history up to that turn
- the reward assigned to that turn depends on its own score and later scores

This is the core sample-efficiency improvement over naive multi-turn RL.

### Reward Design

Each kernel receives a scalar score

- correctness reward: `0.3`
- plus speedup over PyTorch Eager if the kernel is correct

The paper reports that heavier correctness weighting makes the model over-optimize for safe but slow kernels, while zero correctness weighting also hurts learning. The chosen score is therefore a deliberate correctness-performance compromise.

For multi-turn credit assignment, Kevin compares several strategies and ends up with discounted future reward aggregation. The best-performing configuration uses:

- sum aggregation over future turn scores
- discount factor `gamma = 0.4`

This lets an early imperfect kernel be rewarded if it enables a later, faster correct kernel.

### Key Design Choices

| Design Choice | Purpose | Effect |
| --- | --- | --- |
| Summarized CoTs from previous turns | Prevent context explosion | Preserves reasoning trace while fitting longer trajectories |
| Train on every refinement turn | Improve sample efficiency | Avoids wasting useful intermediate rollouts |
| Discounted future reward | Better credit assignment | Rewards exploratory steps that lead to strong later kernels |
| Stronger base model (`QwQ-32B`) | Avoid sparse-reward collapse and reward hacking | Smaller bases failed to learn reliably |
| Strict format and anti-hacking checks | Prevent fake wins | Forces kernels to be pure CUDA instead of partially falling back to PyTorch |

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Training Tasks | 180 KernelBench tasks from Levels 1 and 2 |
| Evaluation Set | 100 tasks: 80 newly constructed + 20 held-out original tasks |
| Base Model | `QwQ-32B` |
| Training Recipe | GRPO with multi-turn or single-turn variants |
| Train-Time Multi-Turn Setup | 16 parallel trajectories, 4 refinement turns |
| Test-Time Main Setup | 16 parallel trajectories, 8 refinement turns |
| Hardware Note | Reported kernel speedups are measured on NVIDIA H200 GPUs |

### Headline Results

| Model | Correctness best@16 | Performance best@16 | `fast_1` best@16 | `fast_1.5` best@16 |
| --- | --- | --- | --- | --- |
| Kevin (Multi-Turn) | `82%` | `1.10x` | `43%` | `20%` |
| Single-Turn RL | `82%` | `0.85x` | `43%` | `16%` |
| QwQ-32B | `56%` | `0.53x` | `23%` | `10%` |
| OpenAI `o4-mini` | `38%` | `0.78x` | `21%` | `13%` |
| OpenAI `o3-mini` | `27%` | `0.30x` | `9%` | `4%` |

Kevin's most important gain is not correctness, since the multi-turn and single-turn RL variants both reach `82%` best@16 correctness. The real difference is performance: `1.10x` versus `0.85x` for single-turn RL, which shows that multi-turn RL is learning better optimization trajectories rather than merely improving compilation or semantic fidelity.

### Inference-Time Scaling

| Model | Inference Config | Performance pass@128 | Correctness pass@128 |
| --- | --- | --- | --- |
| Multi-Turn RL | `16 traj x 8 turns` | `1.10x` | `82%` |
| Multi-Turn RL | `32 traj x 4 turns` | `1.02x` | `83%` |
| Multi-Turn RL | `128 traj x 1 turn` | `0.65x` | `76%` |
| Single-Turn RL | `16 traj x 8 turns` | `0.85x` | `82%` |
| Single-Turn RL | `128 traj x 1 turn` | `0.70x` | `73%` |
| QwQ-32B | `16 traj x 8 turns` | `0.53x` | `57%` |

This table supports one of the paper's strongest claims: for kernel generation, extra sequential refinement is more valuable than spending the same budget on pure parallel sampling. Kevin benefits the most from longer refinement chains, which suggests that training on multi-turn trajectories transfers directly to test-time iterative improvement.

### Training Behavior and Stability

| Observation | Main Finding |
| --- | --- |
| Single-turn reward curve | Plateaus early because near-correct kernels get zero reward and the model becomes conservative |
| Multi-turn reward curve | Keeps increasing because refinement turns expose denser learning signal |
| Reward formulation ablation | Sum aggregation with `gamma=0.4` scales best over 8 turns |
| Instability diagnosis | "Not Okay Ratio" predicts later junk generation before full collapse |
| Stabilization | Gradient norm clipping and constant length normalization delay collapse better than KL regularization |

The "Not Okay Ratio" is one of the paper's more unusual but useful observations. Since `QwQ-32B` normally starts CoTs with "Okay,", weird variations of that opening become an early warning sign that the policy is destabilizing before obvious junk text appears.

## Limitation

The paper is clear that Kevin is a promising training recipe, but the evidence is still bounded by compute, benchmark design, and hardware specificity.

| Limitation | Why It Matters |
| --- | --- |
| Limited RL training length | Training only reaches about 80 gradient steps because long-horizon kernel RL is expensive |
| Model instability remains a real issue | Additional RL on a heavily post-trained base can still cause junk generation and collapse |
| Speedups are tied to fixed task dimensions | KernelBench evaluates predefined tensor shapes, so measured gains are not automatically shape-robust |
| Hardware specificity | Reported speedups are accurate only for the tested dimensions on NVIDIA H200 GPUs |
| Reward hacking still requires active defense | The model may copy PyTorch logic or only partially fuse kernels unless strict format checks are enforced |

---

*Reading date: 2026-04*
*Note status: Completed*
