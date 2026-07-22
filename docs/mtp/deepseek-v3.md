# DeepSeek-V3 Technical Report

<div class="paper-meta" markdown>

**Authors**: DeepSeek-AI  
**Institution**: DeepSeek-AI  
**Conference**: arXiv  
**Year**: 2025  
**Paper Link**: [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)  
**GitHub**: [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Multi-Token Prediction</span>
<span class="paper-tag">Mixture-of-Experts</span>
<span class="paper-tag">LLM Training</span>
</div>

## Background

DeepSeek-V3 is a 671B-parameter Mixture-of-Experts language model with 37B activated parameters per token. The report is broader than MTP: it covers MLA attention, DeepSeekMoE, auxiliary-loss-free load balancing, FP8 training, pipeline scheduling, post-training, and deployment. Its relevance to MTP is that a frontier-scale open model uses MTP as a first-class auxiliary training objective rather than as a small standalone experiment.

The MTP module is designed mainly to improve the main model during training. It can be discarded at inference with no additional main-model cost, or repurposed as a speculative decoding module for latency reduction. This makes DeepSeek-V3 an important engineering data point: MTP can be integrated into very large MoE pretraining without changing the basic autoregressive interface of the final model.

**Key Takeaways**

- DeepSeek-V3 uses sequential MTP modules that preserve a causal chain across prediction depths.
- The MTP loss is auxiliary and weighted; the module can be removed during ordinary inference.
- Ablations show broad benchmark gains, with especially large improvements on HumanEval and GSM8K.

## Methodology

DeepSeek-V3 differs from the parallel independent-head design of Gloeckle et al. Instead of predicting all additional tokens independently from the same trunk state, it uses `D` sequential MTP modules. Each module predicts one additional future token while consuming the previous-depth representation and the embedding of the next ground-truth token. This retains a complete causal chain for each prediction depth.

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Main DeepSeek-V3 model | Token sequence | Main hidden states | Produces the base autoregressive representation |
| MTP projection `M_k` | Previous-depth hidden state plus shifted token embedding | MTP module input | Combines model state with causal shifted-token context |
| MTP transformer block `TRM_k` | Projected representation | Depth-specific hidden state | Performs one sequential future-token prediction step |
| Shared embedding | Ground-truth shifted tokens | Token embeddings | Shares parameters and gradients with the main model |
| Shared output head | MTP hidden states | Future-token distributions | Reuses the main vocabulary projection |
| Weighted MTP loss | Per-depth cross-entropy losses | Auxiliary objective | Densifies training signal without changing AR inference |

### Key Design Choices

- Use a sequential causal design rather than independent parallel heads.
- Share the embedding and output head with the main model to reduce parameter overhead and align gradients.
- Weight the averaged MTP loss by `lambda`; the report uses `lambda = 0.3` for the first 10T pretraining tokens and `0.1` for the remaining 4.8T tokens.
- Keep the MTP module optional at inference: discard it for normal serving, or use it as a speculative drafter.

## Experiment

### Experimental Setup

| Item | Value |
| --- | --- |
| Full model | 671B total parameters, 37B activated per token |
| Pretraining data | 14.8T tokens |
| Training system | 2048 NVIDIA H800 GPUs with FP8 mixed precision |
| MTP ablation models | Small MoE: 15.7B total, 2.4B activated; Large MoE: 228.7B total, 20.9B activated |
| MTP ablation change | Add a 1-depth MTP module, keep training data and other architecture choices fixed |

### Headline Results

| Benchmark | Small MoE Baseline | Small MoE w/ MTP | Large MoE Baseline | Large MoE w/ MTP |
| --- | ---: | ---: | ---: | ---: |
| BBH EM | 39.0 | 41.4 | 70.0 | 70.7 |
| MMLU EM | 50.0 | 53.3 | 67.5 | 66.6 |
| HumanEval Pass@1 | 20.7 | 26.8 | 44.5 | 53.7 |
| GSM8K EM | 25.4 | 31.4 | 72.3 | 74.0 |
| MATH EM | 10.7 | 12.6 | 38.6 | 39.8 |

The ablation shows gains on most benchmarks, with the clearest improvements on reasoning and coding tasks. The large MoE improves HumanEval Pass@1 by 9.2 absolute points. The report also evaluates inference use: DeepSeek-V3 predicts the next two tokens through MTP, and when combined with speculative decoding, the acceptance rate of the second-token prediction is reported at 85%-90% across generation topics, giving 1.8x tokens-per-second improvement.

## Limitation

| Limitation | Why It Matters |
| --- | --- |
| MTP evidence is embedded in a large system report | The gains cannot be fully isolated from all architecture and training-system choices in the final model |
| Ablations use smaller proxy MoE models | They support the design but are not a direct full-scale ablation of the 671B model |
| Sequential MTP adds training complexity | The module shares parameters and preserves causality, but still adds extra transformer blocks and loss scheduling |
| Inference acceleration is briefly reported | The 1.8x TPS result and 85%-90% acceptance are high-level numbers without the full serving study expected in a dedicated decoding paper |

---

*Reading date: 2026-07*  
*Note status: Completed*
