# DFlash: Block Diffusion for Flash Speculative Decoding

<div class="paper-meta" markdown>

**Authors**: Jian Chen, Yesheng Liang, Zhijian Liu  
**Conference**: arXiv  
**Year**: 2026  
**Paper Link**: [arXiv:2602.06036](https://arxiv.org/abs/2602.06036)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Speculative Decoding</span>
<span class="paper-tag">Block Diffusion</span>
<span class="paper-tag">Parallel Drafting</span>
</div>

## Background

Autoregressive speculative drafters improve proposal quality by conditioning every token on earlier draft tokens, but a block of length `gamma` requires `gamma` sequential draft passes and vocabulary projections. Their latency grows with the speculation budget, forcing even strong methods such as EAGLE-3 to use shallow drafters. Diffusion language models can predict masked positions together, yet large pretrained diffusion drafters are expensive and small standalone drafters lack the target model's knowledge.

DFlash treats speculative decoding as a favorable role for diffusion: the draft can be approximate because exact target verification corrects it. A lightweight block diffusion adapter predicts the full candidate block in one pass, while hidden features from the frozen target supply the semantic capacity missing from a small drafter.

**Key Takeaways**

- One-pass block drafting removes the sequential draft latency that limits autoregressive speculation.
- Persistent target-feature injection lets a deeper lightweight drafter scale acceptance without reasoning from scratch.
- Verification remains standard and lossless; diffusion affects proposal efficiency, not the final distribution.

## Methodology

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Target feature extractor | Frozen target forward pass | States from five distributed layers | Supplies multi-level context and future-token information |
| Feature projector | Concatenated target states | Compact context features | Aligns target and draft hidden dimensions |
| KV injection | Context features | Per-layer draft K/V cache | Reintroduces target context at every drafter layer |
| Block diffusion drafter | Anchor token plus masked positions | All draft-token distributions | Predicts a block in one parallel pass |
| Shared embedding and LM head | Draft states | Vocabulary logits | Keeps the adapter aligned while freezing large matrices |
| Target verifier | Draft block | Accepted prefix and bonus token | Recovers the exact target distribution |

KV injection is the main conditioning mechanism. Rather than adding a target feature only to the first draft-layer input, DFlash projects it into the keys and values of every layer and caches it. This prevents context information from being diluted as draft depth grows.

Training samples response positions as anchors, masks the following `block_size - 1` tokens, and predicts each block in parallel. Multiple independent blocks share a packed forward pass with sparse attention: tokens interact bidirectionally within a block but cannot leak information across blocks. Position-weighted cross-entropy emphasizes early tokens because one early rejection discards the entire suffix.

### Key Design Choices

- Use five draft layers for Qwen3 4B/8B and a block size of 16; use eight layers for Qwen3-Coder.
- Draw target features uniformly from shallow through deep layers and inject them into every draft layer.
- Train on roughly 800K Nemotron/Post-Training and CodeAlpaca prompts with target-generated responses.
- Bound long-context training cost by sampling a fixed number of anchor blocks per sequence.

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Target models | Qwen3 4B/8B, Qwen3-Coder-30B-A3B-Instruct, LLaMA-3.1-Instruct 8B |
| Tasks | GSM8K, MATH-500, AIME25, HumanEval, MBPP, LiveCodeBench, MT-Bench, Alpaca |
| Primary hardware | NVIDIA H200; SGLang serving on one B200 with FlashAttention-4 |
| Baseline | Vanilla autoregressive decoding and EAGLE-3 |
| Metrics | Average acceptance length, end-to-end speedup, serving throughput |

### Headline Results

| Setting | Result |
| --- | --- |
| Qwen3 4B/8B, greedy average | About 4.9x speedup, `tau` about 6.5 |
| Qwen3 4B/8B, temperature 1 average | 4.24x / 4.03x speedup |
| Best Transformers result | 6.09x on MATH-500 for Qwen3-4B |
| Reasoning mode | Roughly 4.5x greedy and 3.9x sampled speedup |
| SGLang concurrency 1-32 | Up to 5.1x throughput on Qwen3-8B |

With an equal 16-token budget, DFlash is around 2.4x faster than EAGLE-3 under greedy decoding and also achieves longer accepted prefixes. The gap reflects both fewer sequential draft operations and lower verification overhead than a large EAGLE tree. Long-context tests reveal a separate adaptation requirement: a 4K-trained drafter degrades beyond its training length, while fine-tuning on only 1.6K LongAlign samples substantially restores acceptance through 32K.

## Limitation

| Limitation | Why It Matters |
| --- | --- |
| Independent block positions | Parallel prediction cannot condition on the realized earlier draft tokens, creating incoherent suffix combinations |
| Target-coupled training | Every target model needs generated responses, hidden-feature selection, and a trained adapter |
| Context-length shift | The base 4K drafter loses acceptance on longer contexts without additional adaptation |
| Narrow baseline availability | Several diffusion speculative methods lacked public implementations and were not compared directly |
| Hardware-specific speedups | Results depend on block kernels, FlashAttention backend, concurrency, and verification cost |

---

*Reading date: 2026-07*
*Note status: Completed*
