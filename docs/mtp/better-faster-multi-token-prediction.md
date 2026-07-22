# Better & Faster Large Language Models via Multi-token Prediction

<div class="paper-meta" markdown>

**Authors**: Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Roziere, David Lopez-Paz, Gabriel Synnaeve  
**Institution**: FAIR at Meta, CERMICS Ecole des Ponts ParisTech, LISN Universite Paris-Saclay  
**Conference**: arXiv  
**Year**: 2024  
**Paper Link**: [arXiv:2404.19737](https://arxiv.org/abs/2404.19737)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Multi-Token Prediction</span>
<span class="paper-tag">Auxiliary Loss</span>
<span class="paper-tag">Self-Speculative Decoding</span>
</div>

## Background

Next-token prediction trains a language model to explain the immediate next token under teacher forcing. This objective is simple and scalable, but it gives dense supervision mostly on local continuations. Many generation tasks, especially code and mathematical reasoning, require choosing an early token that stays consistent with several later tokens. A one-step loss can underemphasize those choice points.

Multi-token prediction changes the pretraining objective. At each sequence position, the model predicts the next `n` tokens in parallel using `n` dedicated output heads on top of a shared transformer trunk. The method is presented as an auxiliary training objective: the model can still use the first head for ordinary autoregressive inference, while the extra heads can be used for self-speculative decoding.

**Key Takeaways**

- MTP improves sample efficiency at larger model scales, especially on generative coding benchmarks.
- A memory-aware implementation avoids materializing all future-token logits simultaneously.
- Extra future-token heads provide a built-in draft mechanism, giving up to 3x inference speedup for 4-token prediction models.

## Methodology

The standard next-token loss minimizes `-log P(x[t+1] | x[:t])`. MTP generalizes this by minimizing the sum of cross-entropy losses for `x[t+1] ... x[t+n]` from the same prefix representation. The implementation uses a shared trunk `f_s`, independent future-token heads `f_h_i`, and a shared unembedding `f_u`. The first head remains the normal next-token head.

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Shared transformer trunk | Causal prefix tokens | Hidden representation | Learns the common context representation |
| Future-token heads | Shared hidden state | Head-specific hidden states | Predict different lookahead positions |
| Shared unembedding | Head outputs | Vocabulary distributions | Keeps token projection tied across heads |
| Sequential head backward pass | One head's logits at a time | Accumulated trunk gradients | Reduces peak memory from `O(nV + d)` to `O(V + d)` |
| Self-speculative decoder | Extra head predictions | Candidate future tokens | Uses MTP heads as internal drafter |

### Key Design Choices

- Treat MTP as auxiliary supervision rather than replacing autoregressive inference.
- Compare models at equal parameter count by moving some depth from the trunk into future-token heads.
- Use head-by-head forward/backward scheduling to avoid the vocabulary-logit memory blowup.
- Evaluate both model quality and decoding acceleration, because the same heads can improve learning and serve as draft heads.

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Models | 300M to 13B parameter decoder-only LMs |
| Data | Code and natural-language corpora; token and byte-level variants |
| Main benchmarks | MBPP, HumanEval, APPS, CodeContests, NLP and summarization tasks |
| Metrics | pass@k, benchmark accuracy, ROUGE, accepted tokens, decoding speedup |
| Inference mode | Ordinary AR decoding or self-speculative decoding with extra heads |

### Headline Results

| Setting | Result |
| --- | --- |
| 13B code models | Solves 12% more HumanEval and 17% more MBPP problems than comparable next-token models |
| 7B, 200B code tokens | 4-token prediction improves MBPP pass@1 from 30.0 to 33.8 and HumanEval pass@1 from 22.8 to 24.0 |
| Byte-level 7B model | 8-byte prediction gives large gains, including 67% more MBPP pass@1 solutions |
| Self-speculative decoding | 4-token prediction reaches 3.0x speedup on code and 2.7x on text |
| Larger byte prediction | 8-byte model reports 6.4x decoding speedup |

The benefits are not uniform. Small models can underperform next-token baselines, while larger models gain more consistently. On natural-language multiple-choice and likelihood benchmarks, 2-token prediction is roughly on par with the baseline and 4-token prediction can regress, but generative summarization and math evaluations show more favorable trends.

## Limitation

| Limitation | Why It Matters |
| --- | --- |
| Gains are scale dependent | Small models may not have enough capacity to benefit from extra prediction heads |
| Optimal lookahead varies | Code, byte-level text, and APPS prefer different `n` values |
| Natural-language choice tasks are mixed | MTP does not automatically improve every evaluation style |
| Extra heads complicate architecture | Although training time is controlled, model design and tuning become less minimal |
| Self-speculative speedup depends on acceptance | Future-token heads only accelerate inference when their predictions match the main head often enough |

---

*Reading date: 2026-07*  
*Note status: Completed*
