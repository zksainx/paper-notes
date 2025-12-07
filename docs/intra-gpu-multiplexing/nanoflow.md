# NanoFlow: Towards Optimal Large Language Model Serving Throughput

<div class="paper-meta" markdown>

**Authors**: Kan Zhu, Liangyu Zhao, Yile Gu  
**Institution**: University of Washington  
**Conference**: OSDI 2025  
**Paper Link**: [arXiv:2408.12757](https://arxiv.org/abs/2408.12757)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">LLM Serving</span>
<span class="paper-tag">Intra-Device Parallelism</span>
<span class="paper-tag">GPU Optimization</span>
</div>

## Abstract

NanoFlow is a novel serving framework that exploits intra-device parallelism to achieve optimal LLM serving throughput. By overlapping the usage of heterogeneous resources within a single GPU device, NanoFlow provides 1.91x throughput boost and achieves 68.5% of optimal throughput.

**Key Contributions**:
- Intra-device parallelism framework for LLM serving
- Overlapping heterogeneous resource usage within single GPU
- 1.91x throughput improvement over baseline systems
- Achieves 68.5% of theoretical optimal throughput

## Core Ideas

NanoFlow addresses the inefficiency in current LLM serving systems by exploiting parallelism opportunities within a single GPU device. Unlike previous approaches that focus on inter-device parallelism, NanoFlow carefully orchestrates compute, memory, and communication resources to maximize utilization through fine-grained intra-device scheduling.

---

*Reading date: 2025*
*Note status: Completed*
