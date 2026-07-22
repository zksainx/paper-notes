# How Transformers Learn to Plan via Multi-Token Prediction

<div class="paper-meta" markdown>

**Authors**: Jianhao Huang, Zhanpeng Zhou, Renqiu Xia, Baharan Mirzasoleiman, Weijie Su, Wei Huang  
**Institution**: University of California, Los Angeles; Shanghai Jiao Tong University; University of Pennsylvania; RIKEN Center for Advanced Intelligence Project; The Institute of Statistical Mathematics  
**Conference**: arXiv  
**Year**: 2026  
**Paper Link**: [arXiv:2604.11912](https://arxiv.org/abs/2604.11912)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Multi-Token Prediction</span>
<span class="paper-tag">Planning</span>
<span class="paper-tag">Transformer Theory</span>
</div>

## Background

MTP is often motivated by empirical gains, but the mechanism behind those gains is less clear. This paper studies the question through planning tasks, where the model must determine a global solution before emitting the first answer token. Standard next-token prediction can exploit teacher-forced prefixes and learn local shortcuts instead of the true planning computation.

The core claim is that MTP changes optimization, not just inference. During evaluation, models still generate one token at a time with the normal autoregressive head. The improvement comes from the training signal: predicting several future tokens from the same prefix encourages the network to form representations that already contain information about later consequences.

**Key Takeaways**

- MTP outperforms NTP on synthetic graph path-finding and more realistic Countdown and SAT tasks.
- The paper identifies a reverse-reasoning circuit: attend to the end node first, then trace predecessor nodes backward.
- A gradient decoupling property gives MTP a cleaner signal than pure NTP for discovering this circuit.

## Methodology

The empirical setup compares MTP and NTP under the same inference mode. MTP with lookahead `k` predicts the next `k` tokens in parallel from the same prefix. The architecture follows the independent-head formulation from Gloeckle et al., using a shared transformer backbone and separate output heads. NTP is the special case `k = 1`.

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Planning task generator | Graph, Countdown, or SAT instance | Serialized prompt and solution | Creates tasks requiring future-aware decisions |
| Shared transformer | Teacher-forced prefix | Hidden representation | Learns state used by all prediction heads |
| MTP heads | Shared representation | Multiple future-token distributions | Provide lookahead supervision |
| Standard AR evaluator | Generated prefix | One next token | Tests whether MTP training improves ordinary decoding |
| Disentangled transformer analysis | Star graph task | Attention and gradient conditions | Makes the planning mechanism mathematically tractable |

### Key Design Choices

- Separate the training objective from inference: all models are evaluated with standard next-token generation.
- Use star graphs to expose the Clever Hans shortcut, then binary trees to test whether MTP helps beyond removing that shortcut.
- Analyze a two-layer disentangled transformer where content and position matching can be studied explicitly.
- Prove that MTP can converge to a reverse-reasoning circuit, while NTP gradients can point away from that circuit.

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Synthetic graph tasks | Star graph and binary tree path finding |
| Realistic planning tasks | Countdown and 3-SAT |
| Countdown model/data | 85M parameter model, 1M samples, 100 epochs |
| 3-SAT model/data | 5.3M parameter model, 50K samples, 300 epochs |
| Reporting | Averages over 3 independent runs |

### Headline Results

| Task | NTP | Best Reported MTP |
| --- | ---: | ---: |
| Star graph | 50% across data sizes | 100% with 2-MTP at 0.5M samples |
| Countdown | 60.27 | 64.93 with 7-MTP |
| 3-SAT | 10.40 | 87.47 with 7-MTP |
| Standard 8-layer transformer check | NTP overfits, around 20% test accuracy | 3-MTP reaches 100% train and test accuracy |

The star graph result shows that MTP helps even with only one extra lookahead target. The binary tree experiments reduce the chance that the result is only due to disabling teacher-forced cheating, because every step requires a decision. Countdown and SAT add evidence that the effect carries beyond toy graph serialization.

The theory explains the pattern. Under MTP, a shallow head receives an isolated signal for the future endpoint, encouraging layer 1 to attend to the end node. A deeper head then uses that endpoint-oriented representation to identify the intermediate predecessor. Under pure NTP, gradients from different layers are entangled and can reinforce a local shortcut rather than the reverse plan.

## Limitation

| Limitation | Why It Matters |
| --- | --- |
| Theory uses simplified transformers | The disentangled model is useful for proof but does not cover all behavior of production LLMs |
| Tasks are controlled | Graph, Countdown, and SAT isolate planning but do not represent the full diversity of language reasoning |
| Focus is reasoning, not serving | The paper does not evaluate MTP as an inference acceleration system |
| Future-token quality is task dependent | Larger lookahead helps SAT strongly, while Countdown gains are more modest |

---

*Reading date: 2026-07*  
*Note status: Completed*
