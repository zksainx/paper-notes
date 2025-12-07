# VIDUR: A Large-Scale Simulation Framework for LLM Inference

<div class="paper-meta" markdown>

**Authors**: Microsoft Research Team
**Institution**: Microsoft Research
**Conference**: MLSys 2024
**Paper Link**: [arXiv:2405.05465](https://arxiv.org/abs/2405.05465)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">LLM Inference</span>
<span class="paper-tag">Serving Simulation</span>
<span class="paper-tag">Performance Optimization</span>
</div>

## Abstract

Vidur is a high-fidelity, extensible simulation framework for LLM inference performance that enables studying system performance and capacity planning without requiring expensive GPU access. It achieves less than 9% error in estimating inference latency across multiple models.

**Key Contributions**:
- High-fidelity simulation using experimental profiling and predictive modeling
- Capacity planning and deployment configuration optimization
- Testing new research ideas (scheduling algorithms, speculative decoding)
- Validated on LLaMA2 7/70B, InternLM-20B, and Qwen-72B

## Core Ideas

Vidur combines experimental profiling with predictive modeling to evaluate end-to-end inference performance. The companion tool Vidur-Search automatically identifies cost-effective deployment configurations meeting performance constraints, finding optimal LLaMA2-70B configuration in 1 hour on CPU (vs 42K GPU hours costing $218K for deployment-based exploration).

---

*Reading date: 2024*
*Note status: Completed*
