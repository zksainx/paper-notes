# CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning

<div class="paper-meta" markdown>

**Authors**: Xiaoya Li, Xiaofei Sun, Albert Wang, Jiwei Li, Chris Shum  
**Institution**: DeepReinforce Team  
**Conference**: ICLR  
**Year**: 2026  
**Paper Link**: [arXiv:2507.14111](https://arxiv.org/abs/2507.14111)  
**GitHub**: [deepreinforce-ai/CUDA-L1](https://github.com/deepreinforce-ai/CUDA-L1)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">CUDA Optimization</span>
<span class="paper-tag">Reinforcement Learning</span>
<span class="paper-tag">KernelBench</span>
</div>

## Background

CUDA-L1 studies automatic CUDA optimization rather than kernel synthesis from scratch. The input is a reference PyTorch implementation from KernelBench, and the goal is to generate a faster CUDA implementation that remains executable and correct. This setting is hard because current frontier LLMs rarely improve performance reliably: the paper reports that even strong general models speed up fewer than 20% of KernelBench tasks.

The central claim is that speedup alone can be used as an RL signal, but only if the model is trained to reason comparatively over prior code variants rather than receiving a scalar reward after generation. CUDA-L1 therefore combines staged CUDA specialization with a new contrastive RL loop that feeds prior implementations and their measured performance back into the prompt.

**Key Takeaways**

- CUDA-L1 uses a three-stage pipeline: SFT, self-supervised success filtering, and contrastive RL for speed.
- Contrastive RL lets the model inspect prior CUDA variants with scores, analyze why some are faster, and then generate a new candidate.
- On KernelBench, CUDA-L1 reports `3.12x` mean speedup over the default baseline on A100, with a peak gain of `120x` and high cross-GPU transfer to H100, L40, RTX 3090, and H20.

## Methodology

CUDA-L1 is trained from a general foundation model through progressively harder objectives. The early stages focus on making CUDA code executable and correct; the final stage focuses on runtime improvement. The backbone throughout the paper is `deepseek-v3-671B`.

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Stage 1: SFT via Data Augmentation | KernelBench PyTorch tasks plus successful CUDA code collected from 6 frontier LLMs | CUDA-specialized model | Teaches CUDA syntax, structure, and optimization patterns from verified examples |
| Stage 2: Self-Supervised Learning | Model-generated CUDA candidates | Updated model trained only on successful outputs | Improves executability and correctness before optimizing speed |
| Stage 3: Contrastive RL | Prompt with prior CUDA variants and their speed scores | Faster CUDA candidate plus GRPO update | Uses comparative analysis and reward-based optimization for performance |
| Reward Measurement Pipeline | Reference code and candidate code on dedicated A100 GPUs | Stable speedup reward | Reduces timing noise with randomized execution order, long evaluation windows, variance checks, and verification reruns |
| Reward-Hacking Mitigation | Suspicious high-reward samples, hacking database, adversarial checker | Smoothed and filtered reward | Prevents the policy from exploiting benchmark loopholes instead of producing real speedups |

### Stage 1 and Stage 2

The SFT stage collects 2,105 successful CUDA implementations across 250 KernelBench tasks by querying six existing LLMs, each with up to 20 trials per task and early stopping after two successful samples. These successful samples are used as supervised targets for fine-tuning.

Stage 2 then runs an iterative self-training loop:

- sample CUDA code from the current model
- test executability and correctness
- keep only successful samples
- update the model on those samples

This stage can be viewed as a sparse-reward REINFORCE variant with only positive updates. The paper argues that this is more stable than baseline-subtracted policy-gradient variants because most early samples are failures.

### Contrastive Reinforcement Learning

The main innovation is that reward is used in two ways simultaneously:

- as a scalar for GRPO parameter updates
- as prompt context for future generations

Each contrastive prompt contains:

- the task description
- two previously generated CUDA codes with measured scores
- explicit generation protocol
- restrictions intended to prevent reward hacking

The model must then produce:

- a comparative performance analysis
- a high-level optimization plan
- the new CUDA implementation

This differs from standard RL for code, where the model never directly sees reward information during generation. CUDA-L1 instead turns performance feedback into reasoning context, so the model can analyze which transformations helped and which were harmful.

### Key Design Choices

| Design Choice | Purpose | Effect |
| --- | --- | --- |
| Performance-bucket exemplar sampling | Select high-quality but performance-diverse prior codes | Enables useful comparison instead of near-duplicate prompt context |
| Median-of-buckets reward | Reduce timing noise from GPU variance | Makes GRPO updates more stable |
| Dedicated-GPU evaluation with verification reruns | Prevent spurious large speedups | Improves trustworthiness of reward signals |
| Reward smoothing and adversarial checking | Mitigate reward hacking | Limits over-optimization on benchmark loopholes |
| Separate correctness-first stages before speed RL | Build a viable code generator first | Avoids spending RL budget on mostly uncompilable candidates |

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Benchmark | KernelBench |
| Tasks | 250 total: 100 Level 1, 100 Level 2, 50 Level 3 |
| Target Training GPU | NVIDIA A100 PCIe |
| Backbone | DeepSeek-V3-671B |
| Main Baselines | Default PyTorch reference, `torch.compile`, `torch.compile` reduce-overhead, CUDA Graph |
| Evaluation Window | 20 minutes per task at test time |
| Main Metrics | Mean / max / percentile speedup, success rate, `>1.01x` speedup count |

### Headline Results on A100

| Baseline | Mean Speedup | Max | Median | Success | Speedup Cases |
| --- | --- | --- | --- | --- | --- |
| Default | `3.12x` | `120x` | `1.42x` | `249/250` | `226/250` |
| Torch Compile | `2.77x` | `69.0x` | `1.72x` | `249/250` | `203/250` |
| Torch Compile RO | `2.88x` | `80.1x` | `1.67x` | `249/250` | `200/250` |
| CUDA Graph | `2.81x` | `97.9x` | `1.20x` | `249/250` | `147/229` |

The strongest aggregate result is against the default KernelBench baseline, where CUDA-L1 delivers broad improvements rather than a few isolated wins. The median speedups above `1.2x-1.7x` across all baseline settings show that gains are not driven solely by a tiny number of outliers, although the very large maxima still matter for some tasks.

### By Task Difficulty

| Level | Default Baseline Mean | Key Pattern |
| --- | --- | --- |
| Level 1 | `2.78x` | Strong wins on primitive kernels, but smaller than Level 2 |
| Level 2 | `3.55x` | Best overall category, suggesting fusion-style workloads benefit most |
| Level 3 | `2.96x` | Still strong over default, but harder to beat strong compile baselines |

The paper highlights Level 2 as the best match for the method: there is enough structure for reusable optimization patterns, but tasks are still local enough for code-level reasoning to matter. Level 3 remains harder because framework and graph-level optimizations become stronger baselines.

### Training and Baseline Comparison

| Method | Mean | Max | Median | Success | Speedup Cases |
| --- | --- | --- | --- | --- | --- |
| Vanilla DeepSeek-R1 | `0.88x` | `14.4x` | `0.75x` | `179` | `18` |
| Evolve DeepSeek-R1 | `1.41x` | `44.2x` | `1.17x` | `248` | `162` |
| CUDA-L1 Stage 1 | `1.14x` | `32.7x` | `1.00x` | `240` | `50` |
| CUDA-L1 Stage 1+2 | `1.36x` | `48.3x` | `1.09x` | `247` | `175` |
| CUDA-L1 Stage 1+2+GRPO | `2.41x` | `84.6x` | `1.33x` | `247` | `207` |

This ablation is one of the paper's most important pieces of evidence. SFT alone mostly teaches the model to produce working CUDA. Self-supervised learning further improves coverage. The major speed jump comes from the final RL stage, which indicates that contrastive performance feedback does more than increase correctness.

### Cross-GPU Generalization

| GPU | Mean Speedup | Max | Median | Success |
| --- | --- | --- | --- | --- |
| A100 | `3.12x` | `120x` | `1.42x` | `249/250` |
| H100 | `3.85x` | `368x` | `1.32x` | `250/250` |
| L40 | `3.13x` | `182x` | `1.31x` | `248/250` |
| H20 | `2.38x` | `63.7x` | `1.34x` | `247/250` |

The portability result is notable because training is done for A100. The optimizations still transfer well to other NVIDIA GPUs, and H100 even shows the highest mean and max gains. This suggests the model often learns broadly useful memory, launch, and fusion strategies rather than only memorizing A100-specific parameter settings.

## Limitation

The paper is unusually explicit about reward hacking and evaluation fragility. That is a strength, but it also exposes the current method's operational limits.

| Limitation | Why It Matters |
| --- | --- |
| Reward hacking is substantial | Early RL runs exploited stream timing, lazy evaluation, hyperparameter manipulation, and caching tricks rather than real optimization |
| Reliable reward measurement is expensive | Each candidate may need dedicated GPUs, long timing windows, variance filtering, and verification reruns |
| Success does not imply universally strong speedup | Many kernels are executable and correct, but not all produce meaningful improvement over stronger baselines like CUDA Graph |
| Architecture transfer is imperfect | Cross-GPU portability is strong, but performance varies materially by device, suggesting hardware-specific fine-tuning could still help |
| Evaluation is KernelBench-centered | The evidence is broad within KernelBench, but generalization to production CUDA codebases and larger systems remains to be established |

---

*Reading date: 2026-04*
*Note status: Completed*
