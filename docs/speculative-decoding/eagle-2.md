# EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees

<div class="paper-meta" markdown>

**Authors**: Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang  
**Institution**: Peking University, Microsoft Research, University of Waterloo, Vector Institute  
**Conference**: arXiv  
**Year**: 2024  
**Paper Link**: [arXiv:2406.16858](https://arxiv.org/abs/2406.16858)  
**GitHub**: [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Speculative Decoding</span>
<span class="paper-tag">Dynamic Draft Tree</span>
<span class="paper-tag">Confidence Ranking</span>
</div>

## Background

Tree-based speculative decoding verifies many branches in one target-model pass, but EAGLE and related methods use the same fixed tree for every context. That assumes token acceptance is determined mostly by tree position. Measurements in EAGLE-2 instead show large context-dependent variation: predictable continuations warrant deep expansion, while uncertain contexts benefit from breadth.

The method also finds that EAGLE's token probabilities are sufficiently calibrated to approximate target acceptance rates. EAGLE-2 therefore changes candidate allocation using confidence already produced by the drafter, requiring neither another prediction model nor additional training.

**Key Takeaways**

- Draft-token acceptance is context-dependent, making one static tree structurally inefficient.
- A node's useful score is its full-prefix survival probability, not its local confidence alone.
- Dynamic expansion and global reranking improve EAGLE by 20%-40% while retaining exact verification.

## Methodology

### System Overview

| Stage | Input | Output | Role |
| --- | --- | --- | --- |
| Confidence estimation | EAGLE draft distribution | Per-edge confidence `c_i` | Approximates local acceptance probability |
| Value computation | Node confidence and ancestor path | `V_i = product(c_j)` | Estimates probability that the complete prefix survives |
| Dynamic expansion | Current frontier values | Children of top-k frontier nodes | Spends draft compute on promising contexts |
| Global reranking | All generated nodes | Top-m connected nodes | Recovers valuable shallow nodes skipped during expansion |
| Tree verification | Flattened nodes and ancestor mask | Accepted prefix | Executes one exact target-model verification pass |

At each layer, tree attention expands only the frontier nodes with the largest path values. Multiplying confidences is essential because a strong local token behind a weak ancestor is unlikely to be reached. Once the depth budget is exhausted, all generated nodes are reranked globally. Since child value cannot exceed parent value, selecting top-valued nodes (with shallow nodes winning ties) preserves a connected tree.

The selected tree is flattened for the target model. Its attention mask allows each node to see only its ancestors, preventing information leakage across branches. Standard speculative rejection sampling is unchanged; dynamic construction affects efficiency, not correctness.

### Key Design Choices

- Reuse the existing EAGLE drafter and its probabilities, so EAGLE-2 adds no training cost.
- Separate expansion from final selection: depth-oriented exploration is not necessarily the best verification set.
- Rank by cumulative survival probability to reflect prefix rejection semantics.
- Keep a fixed total verification budget while adapting its topology to each context.

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Target models | Vicuna 7B/13B, LLaMA2-Chat 7B/13B/70B, LLaMA3-Instruct 8B/70B |
| Tasks | MT-Bench, HumanEval, GSM8K, Alpaca, CNN/DailyMail, Natural Questions |
| Baselines | Vanilla decoding, speculative sampling, PLD, Medusa, Lookahead, Hydra, EAGLE |
| Metrics | Wall-clock speedup and average acceptance length `tau` |
| Typical tree setting | 48-60 final tokens, depth 6, top-10 expansion |

### Headline Results

| Setting | Result |
| --- | --- |
| Overall reported range | 3.05x-4.26x speedup, 20%-40% faster than EAGLE |
| Vicuna 13B, temperature 0 mean | 4.04x speedup, `tau = 4.65` |
| LLaMA2-Chat 13B, temperature 0 mean | 4.10x speedup, `tau = 4.68` |
| LLaMA2-Chat 70B on MT-Bench | 3.51x speedup vs. EAGLE's 3.01x |
| HumanEval maximum in main table | 5.00x speedup, `tau = 5.52` |

Across tasks, each cycle advances roughly 4-5.5 tokens. Gains are smaller on Natural Questions and CNN/DailyMail, plausibly because EAGLE-style heads are trained on instruction data with less world knowledge and summarization coverage. On Vicuna-7B MT-Bench, removing both path value and reranking gives 2.81x; adding value reaches 3.48x, and the full method reaches 3.62x with `tau = 4.98`.

## Limitation

| Limitation | Why It Matters |
| --- | --- |
| Depends on calibration | Miscalibrated draft probabilities can allocate tree capacity to the wrong branches |
| Fixed token budget remains | Topology adapts, but the method does not jointly optimize verification size for system load |
| Inherits EAGLE training coverage | Weakly represented domains have shorter accepted prefixes |
| Hardware-dependent speedup | More accepted tokens do not map uniformly to latency across devices and batch sizes |
| Search overhead | Value tracking, reranking, flattening, and mask construction add runtime work |

---

*Reading date: 2026-07*
*Note status: Completed*
