# IC-Cache: Efficient Large Language Model Serving via In-Context Caching

<div class="paper-meta" markdown>

**Authors**: Yifan Yu, Yu Gan, Nikhil Sarda, Lillian Tsai, Jiaming Shen, Yanqi Zhou, et al.  
**Institution**: UIUC; Google; University of Washington  
**Conference**: SOSP '25  
**Year**: 2025  
**Paper Link**: [Author PDF](https://tslilyai.github.io/papers/icc.pdf)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">in-context-learning</span>
<span class="paper-tag">model-routing</span>
<span class="paper-tag">serving-cache</span>
</div>

## Background

Large models offer quality but high latency/cost; small models are cheap but weaker. More than 70% of requests in four studied traces have a semantically similar historical request, but exact response caching has low hit rate and approximate response reuse can return an off-topic answer.

## Methodology

IC-Cache retrieves a high-utility historical request/response produced by a large model and prepends it as an in-context demonstration for a smaller model. It then routes requests across model tiers according to load and predicted quality.

| Mechanism | Role |
| --- | --- |
| Utility-aware retrieval | Select semantically relevant, quality-improving demonstrations |
| Adaptive router | Balance quality, latency, and current serving load |
| Cost-aware replay | Re-run valuable examples offline to improve cached demonstrations |
| Bounded cache policy | Retain/evict examples under drift, cost, and privacy constraints |

## Experiment

| Metric | Result |
| --- | --- |
| Serving throughput | 1.4x-5.9x improvement |
| Latency | 28-71% reduction without measured quality loss |
| Small-model quality | Win rate over larger models improves by up to 12.4 percentage points in reported comparisons |

The evaluation spans millions of realistic queries and integrations with Hugging Face, vLLM, and LangChain.

## Limitation

- In-context examples consume tokens and may expose historical/private content.
- Quality prediction and automatic judging can be noisy under domain drift.
- The method relies on repetition across requests and is weaker on unique tasks.

---

*Reading date: 2026-08*
*Note status: Completed*
