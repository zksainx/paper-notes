# Frontier: Simulating the Next Generation of LLM Inference Systems

<div class="paper-meta" markdown>

**Authors**: Yicheng Feng, Xin Tan, Kin Hang Sew  
**Institution**: CUHK, StepFun  
**Conference**: arXiv 2025  
**Paper Link**: [arXiv:2508.03148](https://arxiv.org/abs/2508.03148)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">LLM Inference</span>
<span class="paper-tag">MoE Models</span>
<span class="paper-tag">Disaggregated Systems</span>
</div>

## Abstract

Frontier is a high-fidelity simulator designed for next-generation LLM inference systems. It handles the growing complexity of Mixture-of-Experts (MoE) models and disaggregated architectures that decouple components like prefill/decode or attention/FFN for heterogeneous scaling.

**Key Contributions**:
- Unified framework for both co-located and disaggregated systems
- Native support for MoE inference with expert parallelism
- Simulation of complex workflows like cross-cluster expert routing
- Advanced pipelining strategies for latency hiding

## Core Ideas

Frontier follows event-driven and modular design principles with a central GlobalController and modular ClusterWorkers. It achieves predicted throughput within 19.0% to 23.2% relative error margin, providing accurate performance trends for system-of-systems nature of modern LLM serving.

---

*Reading date: 2025*
*Note status: Completed*
