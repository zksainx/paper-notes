# Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration

<div class="paper-meta" markdown>

**Authors**: Zejia Lin, Hongxin Xu, Guanyi Chen  
**Institution**: Sun Yat-sen University  
**Conference**: arXiv 2025  
**Paper Link**: [arXiv:2504.19516](https://arxiv.org/abs/2504.19516)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">GPU Utilization</span>
<span class="paper-tag">Spatial-Temporal Orchestration</span>
<span class="paper-tag">Performance Modeling</span>
</div>

## Abstract

Bullet is a novel spatial-temporal orchestration system that eliminates GPU inefficiencies through precise phase coordination. It enables concurrent execution of prefill and decode phases while dynamically provisioning GPU resources using real-time performance modeling. Bullet delivers 1.26x average throughput gains (up to 1.55x) over state-of-the-art systems while consistently meeting latency constraints.

**Key Contributions**:
- Dynamic spatial-temporal GPU resource orchestration
- Concurrent prefill and decode execution
- Real-time performance modeling for resource provisioning
- 1.26x average throughput improvement (up to 1.55x)

## Core Ideas

Bullet addresses the fundamental mismatch between compute-intensive prefill and memory-bound decode phases in LLM serving. The system identifies two key inefficiencies: (1) prefill suffers from suboptimal compute utilization due to wave quantization and attention bottlenecks, and (2) hybrid batches prioritize latency over throughput. Bullet's spatial-temporal orchestration dynamically coordinates phase execution and resource allocation to maximize GPU utilization while meeting latency requirements.

---

*Reading date: 2025*
*Note status: Completed*
