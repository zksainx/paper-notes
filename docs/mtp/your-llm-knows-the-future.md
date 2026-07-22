# Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential

<div class="paper-meta" markdown>

**Authors**: Mohammad Samragh, Arnav Kundu, David Harrison, Kumari Nishu, Devang Naik, Minsik Cho, Mehrdad Farajtabar  
**Institution**: Apple  
**Conference**: arXiv  
**Year**: 2025  
**Paper Link**: [arXiv:2507.11851](https://arxiv.org/abs/2507.11851)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Multi-Token Prediction</span>
<span class="paper-tag">Gated LoRA</span>
<span class="paper-tag">Speculative Decoding</span>
</div>

## Background

Autoregressive LMs are trained and served one token at a time, but their hidden states often contain partial information about later tokens. This paper starts from that observation: if placeholder tokens are appended after a prompt, the correct future tokens already appear surprisingly high in the pretrained model's logits. The model has some future-token knowledge, but it is not organized into a reliable multi-token generator.

The method adapts an existing autoregressive model with minimal retraining. It appends mask tokens, trains only lightweight gated LoRA adapters and a sampler head, and then uses speculative verification to keep generation aligned with the original model. This places the work between MTP and speculative decoding: it trains a future-token capability, then measures its usefulness by how many tokens can be accepted per generation step.

**Key Takeaways**

- Mask tokens expose future-token predictions from a pretrained AR model without rebuilding the model from scratch.
- Gated LoRA preserves original next-token behavior by activating adapters only on MTP tokens.
- Quadratic decoding improves verification robustness and achieves strong speedups on math and coding tasks.

## Methodology

The model appends `k` unique mask tokens to a prefix. Standard NTP positions preserve normal autoregressive behavior, while MTP mask positions learn to predict future tokens. During training, the input, labels, position IDs, and attention biases are modified so many prefix-plus-mask prompts are simulated in one model call.

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Mask-token formulation | Prefix plus `k` masks | Hidden states for NTP and MTP tokens | Turns future-token prediction into supervised fine-tuning |
| Gated LoRA | Decoder activations and token-type gate | MTP-only adapter updates | Prevents changes to ordinary NTP-token outputs |
| Sampler head | Mask hidden state plus previous sampled token | Coherent token logits | Conditions each future token on the prior sampled token |
| Latent consistency loss | NTP hidden states and corresponding MTP hidden states | Alignment penalty | Encourages MTP predictions to match AR predictions |
| Linear decoding | Verified prefix plus previous speculative tokens | Candidate verification | Simple self-speculative verification |
| Quadratic decoding | Speculative tokens interleaved with masks | More robust candidate set | Maintains `k` candidates for future verification |

### Key Design Choices

- Freeze the base Tulu3-8B SFT model and train only LoRA and sampler parameters.
- Gate LoRA by token type, so NTP tokens follow the original model path while mask tokens receive adaptation.
- Use a two-layer MLP sampler to make future-token selections sequentially coherent.
- Add latent consistency matching so MTP states imitate the corresponding NTP states without modifying the NTP anchors.

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Base model | Tulu3-8B SFT |
| Trainable modules | Gated LoRA rank 128 and 2-layer sampler MLP |
| Training | 50K iterations on 8 NVIDIA A100 GPUs |
| Optimizer | AdamW, learning rate `2e-4`, 5K warmup iterations |
| Main metric | Acceptance rate, interpreted as tokens generated per decoding step |
| Domains | Knowledge, math, coding, chat, and safety benchmarks |

### Headline Results

| Benchmark | 1 Mask | 4 Masks | 8 Masks |
| --- | ---: | ---: | ---: |
| MMLU | 1.54 | 2.18 | 2.38 |
| GSM8K | 1.84 | 3.75 | 5.22 |
| HumanEval | 1.86 | 3.87 | 5.35 |
| AlpacaEval | 1.61 | 2.31 | 2.52 |
| Average | 1.67 | 2.69 | 3.17 |

The strongest gains appear on structured domains where future tokens are more predictable: HumanEval reaches 5.35 accepted tokens per step and GSM8K reaches 5.22 with 8 masks. Chat and knowledge tasks still improve, but more modestly, around 2.4x-2.5x for MMLU and AlpacaEval at 8 masks.

Ablations show that quadratic decoding, the sampler head, and latent consistency loss each contribute to speedup. The LoRA-rank study supports the central hypothesis: even very low ranks can extract future-token potential, while the sampler head matters more than simply increasing LoRA rank. Very high ranks can hurt speedup, likely because the limited SFT data makes overfitting easier.

## Limitation

| Limitation | Why It Matters |
| --- | --- |
| Requires fine-tuning | It is lightweight compared with full training, but not a pure inference-only method |
| Gated LoRA cannot be fused normally | The binary gate keeps NTP behavior stable but leaves some adapter overhead at inference |
| Speedup is domain dependent | Coding and math benefit more than open-ended chat or knowledge tasks |
| Quadratic decoding adds parallel work | The paper argues `k` is small relative to context length, but the extra work is still system dependent |
| Quality preservation relies on verification | Naive multi-token generation would degrade output; speculative checking is essential |

---

*Reading date: 2026-07*  
*Note status: Completed*
