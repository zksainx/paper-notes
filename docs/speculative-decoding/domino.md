# Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding

<div class="paper-meta" markdown>

**Authors**: Jianuo Huang, Yaojie Zhang, Qituan Zhang, Hao Lin, Hanlin Xu, Linfeng Zhang  
**Conference**: arXiv  
**Year**: 2026  
**Paper Link**: [arXiv:2605.29707](https://arxiv.org/abs/2605.29707)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Speculative Decoding</span>
<span class="paper-tag">Causal Correction</span>
<span class="paper-tag">Parallel Drafting</span>
</div>

## Background

Autoregressive and parallel drafters occupy opposite sides of the quality-cost trade-off. EAGLE-3 explicitly conditions each proposal on earlier draft tokens and obtains long accepted prefixes, but repeatedly executes its drafter and full LM head. DFlash produces a complete block in one pass, sharply reducing latency, but each position marginalizes over possible predecessors and can combine incompatible modes.

Domino separates causal modeling from expensive autoregressive execution. A deep parallel backbone still runs once for the whole block; a small sequential correction module then conditions each position on the already sampled prefix. The costly representation learning remains parallel while only a low-rank logit residual is sequential.

**Key Takeaways**

- Intra-block causality can be restored without rerunning the draft Transformer for every token.
- A low-rank correction reuses base vocabulary logits and avoids another full LM-head projection.
- A base-anchored curriculum prevents teacher-forced causal correction from weakening the underlying parallel drafter.

## Methodology

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Parallel draft backbone | Target context and masked block | Preliminary states for all positions | Performs the expensive computation once |
| Frozen LM head | Preliminary states | Base logits `u_k` | Produces independent parallel distributions |
| Causal encoder | Previously sampled draft tokens | Prefix state | Summarizes the realized intra-block path |
| Low-rank correction head | Prefix state and base state | Residual vocabulary logits | Adds causal information cheaply |
| Sequential sampler | Corrected distributions | Draft block | Samples left-to-right without rerunning the backbone |
| Target verifier | Corrected draft probabilities | Accepted prefix | Applies exact speculative verification |

The Domino head uses a compact GRU-like causal encoder and a rank-256 correction projection. At each position, corrected logits are the base logits plus a prefix-dependent residual. This costs much less than recomputing five draft layers and a dense full-vocabulary head at every step.

Training uses the gold prefix to stabilize causal encoding, but unrestricted teacher forcing can let the correction branch dominate before the base distribution becomes useful. The base-anchored curriculum first emphasizes the parallel backbone and gradually shifts optimization toward the corrected distribution. Runtime kernels fuse sampling and correction operations and retain the block-parallel backbone.

### Key Design Choices

- Keep a five-layer DFlash-like backbone and a 16-token block for all Qwen3 targets.
- Use a hidden size of 1024 in the causal encoder and 256 in the low-rank correction.
- Add corrections residually so the parallel distribution remains a stable anchor.
- Train with target-generated data; evaluate both Transformers latency and SGLang throughput.

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Target models | Qwen3 4B and 8B |
| Tasks | GSM8K, MATH, AIME25, HumanEval, MBPP, LiveCodeBench, MT-Bench, Alpaca |
| Baselines | EAGLE-3, DART, DFlash, vanilla autoregressive decoding |
| Hardware | NVIDIA A100-SXM4-80GB |
| Metrics | Acceptance length, end-to-end speedup, SGLang TPS |

### Headline Results

| Setting | Result |
| --- | --- |
| Greedy average, Qwen3-4B | 5.47x vs. DFlash 4.70x |
| Greedy average, Qwen3-8B | 5.49x vs. DFlash 4.66x |
| Temperature 1 average | 4.61x (4B) and 4.46x (8B) |
| Best benchmark preview | 7.92x on GSM8K, vs. DFlash 5.21x |
| SGLang high concurrency | Up to 5.8x throughput speedup |

The Domino head adds 56M parameters, about 5.3% over the parallel drafter, and increases total draft-plus-verify latency by only 2.8% in the reported breakdown. In return, acceptance length rises 16.6% and end-to-end speedup 12.3%. Disabling the head reduces average acceptance from 4.19 to 3.49 and speedup from 3.31x to 2.84x in the controlled ablation.

## Limitation

| Limitation | Why It Matters |
| --- | --- |
| Training cost is not reduced | Target-aligned data and finetuning remain necessary even though inference is faster |
| SGLang-centered implementation | Compatibility and kernel quality in other serving frameworks are not systematically established |
| Residual sequential loop | The correction head is cheap but still introduces latency that grows with block length |
| Limited model-family evidence | Main experiments cover only Qwen3 4B and 8B |
| Platform sensitivity | Memory bandwidth, GPU compute, and kernel implementation change realized speedup |

---

*Reading date: 2026-07*
*Note status: Completed*
