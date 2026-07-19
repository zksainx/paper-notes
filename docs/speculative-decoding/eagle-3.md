# EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test

<div class="paper-meta" markdown>

**Authors**: Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang  
**Institution**: Peking University, Microsoft Research, University of Waterloo, Vector Institute  
**Conference**: arXiv  
**Year**: 2025  
**Paper Link**: [arXiv:2503.01840](https://arxiv.org/abs/2503.01840)  
**GitHub**: [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Speculative Decoding</span>
<span class="paper-tag">Training-Time Test</span>
<span class="paper-tag">Feature Fusion</span>
</div>

## Background

EAGLE constrains each draft hidden state to regress the target model's top-layer feature. This gives a useful intermediate target, but the feature is optimized for the immediate next-token logits and constrains representations needed for farther-ahead prediction. Increasing EAGLE's training data consequently produces diminishing acceptance gains.

Simply deleting feature regression creates a train-test mismatch: later draft steps consume the model's own unconstrained outputs, states that never appear in one-step teacher-forced training. EAGLE-3 addresses both issues by directly optimizing token prediction and explicitly simulating multi-step drafting during training, termed training-time test (TTT).

**Key Takeaways**

- Feature regression is an unnecessary bottleneck when accepted tokens are the actual objective.
- Training on self-generated intermediate states makes unconstrained multi-step drafting stable.
- Fusing low-, middle-, and high-layer target features exposes richer future-token information and restores data scaling.

## Methodology

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Multi-layer tap | Frozen target low/mid/high states | Concatenated target feature | Captures lexical, semantic, and predictive information |
| Feature fusion | Concatenated feature | Hidden-size context `g` | Compresses target information with an FC layer |
| Token conditioning | `g` and latest sampled-token embedding | Draft hidden state `a` | Resolves sampling uncertainty |
| Draft decoder | Fused sequence | Direct token distributions | Uses one decoder layer without feature regression |
| Training-time test | Teacher states plus prior draft predictions | Multi-step rollout examples | Matches recursive inference inputs during training |
| Dynamic tree | Draft probabilities | Context-aware candidates | Reuses EAGLE-2 expansion and verification |

Training first performs a normal prediction, then inserts the produced draft representation into later simulated steps under a custom causal mask. Token loss is applied across these rollout positions. Because outputs no longer need to match a unique target hidden feature, the draft representation can specialize for multi-token prediction and accept arbitrary feature inputs.

At inference, target features from three depths are fused and paired with the embedding of the realized sampled token. The drafter produces tokens autoregressively and EAGLE-2 constructs a context-aware tree. Exact target verification remains unchanged.

### Key Design Choices

- Remove Smooth L1 feature regression and train directly toward token distributions.
- Expose model-generated states during training instead of relying only on clean target features.
- Fuse multiple target layers rather than the top layer alone.
- Scale from roughly 68K ShareGPT samples to ShareGPT plus 464K UltraChat entries; use target-generated responses for alignment.

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Target models | Vicuna 13B, LLaMA-3.1-Instruct 8B, LLaMA-3.3-Instruct 70B, DeepSeek-R1-Distill-LLaMA 8B |
| Tasks | MT-Bench, HumanEval, GSM8K, Alpaca and reasoning evaluations |
| Baselines | Vanilla, speculative sampling, PLD, Medusa, Lookahead, Hydra, HASS, EAGLE, EAGLE-2 |
| Training | AdamW, learning rate `5e-5`, ShareGPT and UltraChat-200K; OpenThoughts for the reasoning model |
| Serving frameworks | SGLang on H100 and vLLM on A100 |

### Headline Results

| Setting | Result |
| --- | --- |
| End-to-end latency | Approximately 3.0x-6.5x over vanilla decoding |
| Relative to EAGLE-2 | 20%-40% faster; about 1.4x at batch size 1 |
| HumanEval | Up to 6.5x speedup and `tau` up to 7.5 |
| SGLang, batch size 64 | 1.38x throughput improvement |
| Training-data scaling | Acceptance and speed continue rising with more data, unlike EAGLE |

The ablation attributes gains to both relaxed representation learning and multi-layer fusion. Code remains easiest to speculate because fixed templates yield predictable continuations. Results on the distilled reasoning model are shaped by domain-specific OpenThoughts training, so its unusually strong GSM8K behavior should not be read as domain-neutral scaling evidence.

## Limitation

| Limitation | Why It Matters |
| --- | --- |
| Higher data-generation cost | Target-generated responses and multi-step training are more expensive than EAGLE's fixed-data training |
| Per-target checkpoints | Feature layers, fusion, and drafter weights remain tied to a particular target model |
| Limited frontier-scale evidence | GPU constraints prevent evaluation on 405B and 671B targets |
| Framework sensitivity | Throughput gains differ between SGLang, vLLM, hardware, and batch sizes |
| Domain-conditioned results | Reasoning gains partly depend on specialized math training data |

---

*Reading date: 2026-07*
*Note status: Completed*
