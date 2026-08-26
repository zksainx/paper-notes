# EcoServe: Efficient LLM Serving on Commodity GPU Clusters

<div class="paper-meta" markdown>

**Authors**: Jiangsu Du, Hongbin Zhang, Taosheng Wei, Zhenyi Zheng, Jiazhi Jiang, Kaiyi Wu, Zhiguang Chen, Yutong Lu  
**Institution**: Sun Yat-Sen University  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/du)  
**GitHub**: [MLSysU/EcoServe](https://github.com/MLSysU/EcoServe)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">commodity-cluster</span>
<span class="paper-tag">partial-disaggregation</span>
<span class="paper-tag">serving</span>
</div>

## Background

Co-located prefill/decode interfere, while full disaggregation requires fast KV transfer unavailable on L20/Ethernet clusters.

## Methodology

PaDG separates prefill/decode in time within an instance, avoiding KV transfer. Several instances form a macro-instance and rotate so prefill remains available. Adaptive routing selects admission/chunk size; “mitosis” scales capacity incrementally.

## Experiment

On 32 L20 GPUs over Ethernet for 30B/70B models, EcoServe improves goodput by 1.96x, 1.99x, 2.51x, and 2.40x over vLLM, Sarathi, DistServe, and MoonCake, and remains competitive on H100/NVLink/InfiniBand.

## Limitation

- Cyclic activation needs enough replicas to cover phase gaps.
- Time separation can increase queueing under skewed prefill/decode mixes.
- Macro-instance scheduling is more complex than independent replicas.

---

*Reading date: 2026-08*
*Note status: Completed*
