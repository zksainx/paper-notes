# Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads

<div class="paper-meta" markdown>

**Authors**: Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao  
**Institution**: Princeton University, Together AI, University of Illinois Urbana-Champaign, Carnegie Mellon University, University of Connecticut  
**Conference**: ICML  
**Year**: 2024  
**Paper Link**: [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)  
**GitHub**: [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Multi-Token Prediction</span>
<span class="paper-tag">Decoding Heads</span>
<span class="paper-tag">Tree Attention</span>
</div>

## Background

Medusa is primarily an inference acceleration framework, but it belongs in the MTP category because its draft source is a set of future-token decoding heads trained on the target model's hidden states. Instead of maintaining a separate draft model, Medusa appends lightweight heads to the backbone LM. Each head predicts a farther future token, and a tree attention verifier checks multiple candidate continuations in one pass.

This design addresses two practical weaknesses of classical speculative decoding. A separate draft model may be unavailable, misaligned with the target model, or awkward to serve in distributed systems. Medusa keeps the drafting mechanism attached to the target model, making it easier to integrate and to train with parameter-efficient methods.

**Key Takeaways**

- Multiple decoding heads turn a single target LM into its own future-token drafter.
- Tree attention verifies many candidate continuations concurrently without expanding batch size.
- Medusa-1 gives lossless acceleration with a frozen backbone; Medusa-2 trains the backbone and heads together for higher speedup.

## Methodology

Medusa follows the speculative decoding loop: generate candidates, process candidates, and accept a valid prefix. Candidate generation uses `K` Medusa heads attached to the last hidden state. The original LM head predicts token `t+1`; the `k`-th Medusa head predicts token `t+k+1`. Each head is a single feed-forward layer with a residual connection, initialized so its output distribution starts close to the original LM head.

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Medusa heads | Last hidden state `h_t` | Future-token distributions | Draft multiple later tokens without a separate model |
| Tree construction | Top predictions from each head | Candidate token tree | Enumerates multiple continuations with a fixed node budget |
| Tree attention | Candidate tree and prefix KV cache | Parallel candidate logits | Verifies candidates in a single target-model-style pass |
| Acceptance rule | Candidate logits | Longest accepted prefix | Uses rejection sampling or typical acceptance |
| Medusa-1 training | Frozen backbone, training data | Trained heads | Adds lossless acceleration with minimal memory |
| Medusa-2 training | Backbone plus heads | Better aligned heads | Improves speedup while preserving next-token quality |

### Key Design Choices

- Use up to five future-token heads; redundant heads can be ignored at inference.
- Weight farther-head losses by a decaying factor such as `0.8^k`.
- Train Medusa-1 when the backbone must remain unchanged; train Medusa-2 when joint fine-tuning is acceptable.
- Build sparse optimized trees from head top-k accuracies, because dense Cartesian-product trees add compute overhead quickly.
- Offer typical acceptance as a practical alternative to exact rejection sampling when matching the exact distribution is less important than plausible high-quality output.

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Models | Vicuna-7B, Vicuna-13B, Vicuna-33B, Zephyr-7B |
| Main benchmark | MT-Bench, with GPT-4 judging quality |
| Training data | ShareGPT, UltraChat, or self-distilled model outputs |
| Shared setup | 5 Medusa heads, 1 layer per head, Axolotl training |
| Main metrics | Acceleration rate, per-step overhead, wall-clock speedup, MT-Bench quality |

### Headline Results

| Model | Acceleration Rate | Overhead | MT-Bench Quality Change | Medusa Speedup |
| --- | ---: | ---: | ---: | ---: |
| Vicuna-7B | 3.47 | 1.22 | +0.01 | 2.83x |
| Zephyr-7B | 3.14 | 1.18 | -0.07 | 2.66x |
| Vicuna-13B | 3.51 | 1.23 | -0.14 | 2.83x |
| Vicuna-33B | 3.01 | 1.27 | +0.05 | 2.35x |

The Vicuna-7B fine-tuning comparison shows the training recipe matters. Medusa-1 reaches 2.18x speedup while preserving quality, and Medusa-2 improves to 2.83x. Directly fine-tuning the model with heads degrades quality, so Medusa-2 uses a combined loss, differential learning rates, and warmup to preserve next-token behavior.

Ablations report an incremental path: Medusa-1 heads without tree attention reach about 1.5x, adding tree attention reaches about 1.9x, optimized tree configuration reaches about 2.2x, and Medusa-2 training reaches about 2.8x. Appendix AlpacaEval results are similar, with Vicuna-13B reaching 3.16x speedup.

## Limitation

| Limitation | Why It Matters |
| --- | --- |
| Exact distribution preservation depends on acceptance mode | Typical acceptance improves practical speed but does not aim for exact rejection-sampling equivalence |
| Best results target batch size 1 | Serving behavior at large batch sizes can differ because compute-bound work grows with candidate count |
| Joint training is delicate | Medusa-2 needs loss balancing, learning-rate separation, and warmup to avoid quality degradation |
| Tree size has diminishing returns | More candidate nodes improve acceptance only until parallel compute overhead dominates |
| Heads are model specific | Each target model needs its own Medusa heads or self-distillation process |

---

*Reading date: 2026-07*  
*Note status: Completed*
