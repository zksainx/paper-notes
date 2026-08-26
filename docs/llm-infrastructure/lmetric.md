# Simple Is Better: Multiplication May Be All You Need for LLM Request Scheduling

<div class="paper-meta" markdown>

**Authors**: Dingyan Zhang, Jinbo Han, Kaixi Zhang, Xingda Wei, Sijie Shen, Chenguang Fang, Wenyuan Yu, Jingren Zhou, Rong Chen  
**Institution**: Shanghai Jiao Tong University; Alibaba Group  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/zhang-dingyan)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">request-routing</span>
<span class="paper-tag">prefix-cache</span>
<span class="paper-tag">load-balancing</span>
</div>

## Background

Cluster routing must prefer instances holding reusable KV state without overloading them. Linear weighted scores need workload-specific tuning or a simulator.

## Methodology

LMETRIC multiplies two indicators: new prefill tokens after KV hits and current instance batch size. Algebraically, comparison cancels the implicit weights, producing a tuning-free balance between reuse and load. The paper derives rare failure conditions and detects them before applying a fallback.

## Experiment

| Baseline | Result |
| --- | --- |
| vLLM-v1 chatbot trace | Mean TTFT -92%, mean TPOT -24% |
| Production scheduler | Mean TTFT -39%, mean TPOT -51% |
| Tail in a reported setting | TTFT -75.6%, TPOT -79.7% |

LMETRIC is deployed in production and evaluated on chatbot and coding-agent traces.

## Limitation

- Batch size and uncached-token count are proxies, not full latency models.
- Rare multiplicative-score failure conditions still need detection/fallback.
- Local scheduler behavior and disaggregated serving can alter the indicators.

---

*Reading date: 2026-08*
*Note status: Completed*
