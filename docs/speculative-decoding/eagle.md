# EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty

<div class="paper-meta" markdown>

**Authors**: Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang  
**Institution**: Peking University, Microsoft Research, University of Waterloo, Vector Institute  
**Conference**: arXiv  
**Year**: 2024  
**Paper Link**: [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)  
**GitHub**: [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Speculative Decoding</span>
<span class="paper-tag">Feature Prediction</span>
<span class="paper-tag">Tree Attention</span>
</div>

## Background

Speculative sampling reduces autoregressive LLM latency by letting a cheap draft model propose several tokens and using one target-model pass to verify them. The acceleration is lossless because rejection sampling exactly recovers the target distribution, but it depends on a drafter that is both cheap and well aligned. A smaller model from the same family is often unavailable or expensive enough to erase the benefit, while lightweight token heads have lower acceptance.

EAGLE starts from two observations. Hidden features immediately before the target LM head are easier to extrapolate than discrete tokens, but their next state is ambiguous unless the sampled token is known. For example, two valid sampled continuations lead to different future features even when the preceding feature is identical. EAGLE removes this uncertainty by supplying the sampled token sequence shifted one step ahead.

**Key Takeaways**

- Feature-level autoregression gives a lightweight drafter access to the target model's representation space.
- Shifted sampled tokens resolve the stochastic ambiguity that limits feature prediction.
- Tree drafting and exact verification advance roughly 3.2-4.5 tokens per target-model pass without changing the output distribution.

## Methodology

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Frozen embedding | Shifted sampled tokens | Token embeddings | Exposes the realized sampling path |
| Fusion layer | Token embeddings and target features | Hidden-size vectors | Combines discrete outcomes with semantic state |
| Autoregression head | Fused history | Predicted next feature | Uses one FC layer and one decoder layer |
| Frozen LM head | Predicted feature | Draft-token distribution | Reuses the target model's vocabulary projection |
| Tree drafter | Draft distributions | Multi-branch token tree | Produces more candidates than draft forward passes |
| Target verifier | Flattened tree and tree mask | Accepted prefix plus bonus token | Preserves the target distribution exactly |

The trainable head predicts the next second-to-top-layer feature from previous target features and the one-step-ahead token sequence. Its objective combines Smooth L1 feature regression with cross-entropy between the target and predicted-feature token distributions: `L = L_reg + 0.1 L_cls`. Uniform feature noise in `[-0.1, 0.1]` is added during training to reduce exposure bias from recursively predicted features.

Tree attention expands several likely branches at each draft step and prevents tokens on different branches from attending to one another. Verification applies multi-round speculative sampling over the candidates. No acceptance threshold is relaxed, so both greedy and sampled generation retain the original target distribution.

### Key Design Choices

- Reuse the target embedding and LM head; train only a small feature autoregression module.
- Condition on the actual sampled token rather than attempting to regress a multimodal future feature.
- Train on 68K ShareGPT conversations; target-generated responses provide only marginal gains in the reported ablation.
- Use a tree rather than a chain to trade additional tokens per pass for longer accepted prefixes.

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Target models | Vicuna 7B/13B/33B, LLaMA2-Chat 7B/13B/70B, Mixtral 8x7B Instruct |
| Tasks | MT-Bench, HumanEval, GSM8K, Alpaca |
| Metrics | Wall-clock speedup, average acceptance length, acceptance rate, throughput |
| Training | 68K ShareGPT dialogues; AdamW; learning rate `3e-5` |
| Trainable size | 0.24B-0.99B parameters across dense target sizes |

### Headline Results

| Setting | Result |
| --- | --- |
| LLaMA2-Chat 70B | 2.7x-3.5x latency speedup and about 2x throughput |
| LLaMA2-Chat 13B, temperature 0 | 3.01x-3.76x speedup across tasks |
| Tokens advanced per target pass | Approximately 3.2-4.5 |
| EAGLE plus gpt-fast on RTX 3090 | 160.4 tokens/s for LLaMA2-Chat 7B |
| Tree attention ablation | Adds 0.6-0.8 accepted tokens and about 0.3x-0.5x speedup |

Code generation benefits most because repeated templates make future tokens easier to draft. The shifted-token ablation moves the reported Vicuna-7B MT-Bench speedup from about 1.9x for feature-only prediction to 2.8x. Tree attention is useful but not solely responsible: chain-only EAGLE still reaches roughly 2.3x-2.7x.

## Limitation

| Limitation | Why It Matters |
| --- | --- |
| Batch-size sensitivity | Speedup falls as larger batches consume the spare GPU compute used for parallel verification |
| Recursive feature error | Predicted features drift from target features; noise training mitigates but does not eliminate this |
| Per-target training | Each target model needs a compatible trained autoregression head |
| MoE behavior | Mixtral reaches only about 1.5x because verifying multiple tokens activates more experts |
| Extra memory | Draft state and tree attention reduce the maximum feasible batch size relative to vanilla decoding |

---

*Reading date: 2026-07*
*Note status: Completed*
