# Optimizing SLO-oriented LLM Serving with PD-Multiplexing

<div class="paper-meta" markdown>

**Authors**: Weihao Cui, Yukang Chen, Han Zhao  
**Institution**: Shanghai Jiao Tong University, Huawei Cloud  
**Conference**: arXiv 2025  
**Paper Link**: [arXiv:2504.14489](https://arxiv.org/abs/2504.14489)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">SLO Optimization</span>
<span class="paper-tag">PD Multiplexing</span>
<span class="paper-tag">GPU Partitioning</span>
</div>

## Abstract

This paper presents Drift, an LLM serving framework that resolves system tensions via PD (Prefill-Decode) multiplexing, enabling in-place and phase-decoupled compute partition. Drift leverages low-level GPU partitioning techniques to multiplex prefill and decode phases spatially and adaptively on shared GPUs while preserving in-place memory sharing.

**Key Contributions**:
- Drift framework for SLO-oriented LLM serving
- In-place and phase-decoupled compute partitioning
- Spatial and adaptive multiplexing of prefill/decode phases
- Low-level GPU partitioning with memory sharing preservation

## Core Ideas

Drift addresses the challenge of meeting Service Level Objectives (SLOs) in LLM serving by introducing PD-multiplexing. The system uses low-level GPU partitioning to spatially multiplex compute-intensive prefill operations with memory-bound decode operations on the same GPU, adaptively adjusting resource allocation based on workload characteristics while maintaining efficient memory sharing.

---

*Reading date: 2025*
*Note status: Completed*
