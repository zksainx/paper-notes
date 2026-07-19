# DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation

<div class="paper-meta" markdown>

**Authors**: Xin Cheng et al.  
**Institution**: Peking University, DeepSeek-AI  
**Conference**: arXiv  
**Year**: 2026  
**Paper Link**: [arXiv:2607.05147](https://arxiv.org/abs/2607.05147)  
**GitHub**: [deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Speculative Decoding</span>
<span class="paper-tag">Semi-Autoregressive Drafting</span>
<span class="paper-tag">Serving Scheduler</span>
</div>

## Background

Parallel drafters can afford deeper networks and predict a long block in one pass, giving them high first-token accuracy. Their positions are independent, however, so later proposals cannot condition on the actual earlier samples. This causes suffix decay and multimodal collisions. Long blocks create a second systems problem: verifying low-confidence suffixes consumes scarce target-model batch capacity under high concurrency even when those tokens will be rejected.

DSpark jointly addresses draft quality and verification allocation. It adds a tiny sequential transition model after a parallel backbone, then schedules a different verification prefix for each request using calibrated survival probabilities and a measured engine-throughput curve.

**Key Takeaways**

- Deep parallel capacity is most valuable at the first position; lightweight sequential dependence stabilizes later positions.
- Verification length is a load-dependent resource-allocation decision, not only a confidence threshold.
- Production scheduling must preserve non-anticipation: using future sampled tokens in admission decisions biases the output distribution.

## Methodology

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Parallel backbone | Target features and draft block | Base logits for all positions | Performs high-capacity computation in one pass |
| Markov/RNN head | Base logits and sampled prefix | Causal logit bias | Mitigates suffix decay with cheap sequential correction |
| Confidence head | Backbone state and previous-token embedding | Conditional acceptance `c_k` | Predicts whether token `k` survives given its prefix |
| Sequential temperature scaling | Validation outcomes | Calibrated prefix probabilities | Corrects cumulative overconfidence without changing rank |
| Hardware profiler | Verification batch size | Steps-per-second lookup | Captures engine-specific capacity cliffs |
| Prefix scheduler | Survival probabilities, active requests, capacity curve | Per-request verification lengths | Maximizes expected system token throughput |

The default sequential module is a rank-256 Markov transition: the previous token selects a low-rank vocabulary bias added to each base distribution. An RNN variant sees the full block prefix but gives only marginal extra gains and is harder to deploy. Training combines position-weighted token cross-entropy, total-variation distribution matching, and confidence BCE with weights `0.1`, `0.9`, and `1.0`.

For request `r`, the probability that position `j` survives is the cumulative product of calibrated conditional confidences. The scheduler globally orders feasible prefix extensions by marginal survival probability and evaluates expected throughput `Theta = expected accepted tokens x SPS(batch size)`. An early stop maintains causal admission in the theoretical algorithm. Production uses an asynchronous two-step-old capacity estimate so CUDA graphs and zero-overhead scheduling remain compatible, while current confidences still determine which tokens are prioritized.

### Key Design Choices

- Preserve exact softmax token probabilities so standard rejection sampling remains lossless.
- Calibrate cumulative prefix survival sequentially; raw confidence has adequate ranking but is overconfident.
- Treat hardware throughput as a profiled discrete function rather than a constant verification cost.
- Flatten variable-length verification tokens physically and encode logical dependencies through sparse-attention markers.

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Offline targets | Qwen3 4B/8B/14B and Gemma4-12B |
| Baselines | Retrained Eagle3 and DFlash using identical data and feature layers |
| Training data | 1.3M Open-PerfectBlend prompts with target-generated responses |
| Tasks | Three math, three code, and three chat benchmarks; temperature 1 |
| Production | DeepSeek-V4-Flash and V4-Pro live serving traffic |

### Headline Results

| Setting | Result |
| --- | --- |
| Accepted length vs. Eagle3 on Qwen3 | +26.7% to +30.9% macro-average |
| Accepted length vs. DFlash on Qwen3 | +16.3% to +18.4% macro-average |
| Longer blocks (`gamma` 15) vs. DFlash | +30% math, +26% code, +22% chat |
| V4-Flash at 80 tok/s/user SLA | +51% aggregate throughput vs. MTP-1 |
| V4-Pro at 35 tok/s/user SLA | +52% aggregate throughput vs. MTP-1 |
| Matched throughput | Per-user speed +60%-85% (Flash), +57%-78% (Pro) |

Offline tests disable scheduling to isolate draft quality. DSpark consistently wins across target scales and domains; for Qwen3-4B it reaches accepted lengths 6.11/5.70/4.89 on GSM8K/MATH/AIME25 and 3.64/3.54/3.29 on the three chat sets. Confidence pruning raises acceptance from 45.7% to 95.7% on chat in a threshold sweep. Sequential temperature scaling reduces cumulative ECE from roughly 3%-8% to about 1%.

Strict-SLA relative gains of 661% and 406% occur where MTP-1 approaches a low-concurrency capacity cliff; the paper appropriately treats these as frontier-extension evidence rather than representative multiplicative speedups. Moderate-SLA and matched-throughput comparisons are more stable.

## Limitation

| Limitation | Why It Matters |
| --- | --- |
| Fixed full-block draft cost | Low-acceptance queries pay for the entire parallel backbone even when verification prunes most tokens |
| Profiler and load assumptions | Scheduling quality depends on an engine-specific throughput table and relatively balanced context lengths |
| Complex serving integration | Dynamic lengths require asynchronous scheduling, sparse markers, and architecture-specific kernel changes |
| Production evidence is proprietary | Live-traffic distributions and DeepSeek-V4 internals limit full external reproduction |
| Calibration drift | Domain or load shifts can make stored survival calibration and scheduling estimates stale |

---

*Reading date: 2026-07*
*Note status: Completed*
