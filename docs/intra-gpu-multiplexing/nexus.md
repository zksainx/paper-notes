# Nexus: Proactive Intra-GPU Disaggregation of Prefill and Decode in LLM Serving

<div class="paper-meta" markdown>

**Authors**: Xiaoxiang Shi, Colin Cai, Junjia Du, Zhihao Jia  
**Institution**: Carnegie Mellon University  
**Conference**: arXiv 2025  
**Paper Link**: [arXiv:2507.06608](https://arxiv.org/abs/2507.06608)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Prefill-Decode Disaggregation</span>
<span class="paper-tag">GPU Multiplexing</span>
<span class="paper-tag">Dynamic Workload</span>
</div>

## Abstract

Nexus achieves proactive intra-GPU disaggregation that adapts effectively to dynamic workloads by managing the conflicting resource demands of prefill and decode phases under varying conditions. It achieves up to 2.2× lower time-between-tokens (TBT) and 1.4× higher throughput than vLLM-disaggregation with only half the number of GPUs.

**Key Contributions**:
- Proactive intra-GPU disaggregation framework
- Adaptive resource management for prefill and decode phases
- 2.2× lower TBT compared to vLLM-disaggregation
- 1.4× higher throughput with 50% fewer GPUs

## Core Ideas

Nexus extends vLLM with fine-grained resource partitioning to enable efficient coexistence of prefill and decode operations on the same GPU. The key insight is managing the dynamic and conflicting resource demands proactively rather than reactively, allowing the system to adapt to varying workload characteristics while maintaining high throughput and low latency.

---

*Reading date: 2025*
*Note status: Completed*
