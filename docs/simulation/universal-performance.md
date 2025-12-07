# Towards Universal Performance Modeling for Machine Learning Training on Multi-GPU Platforms

<div class="paper-meta" markdown>

**Authors**: Research Team
**Institution**: Multiple Institutions
**Conference**: arXiv 2024, IEEE
**Paper Link**: [arXiv:2404.12674](https://arxiv.org/abs/2404.12674)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Performance Modeling</span>
<span class="paper-tag">Multi-GPU</span>
<span class="paper-tag">Distributed Training</span>
</div>

## Abstract

This paper addresses characterizing and predicting training performance of modern ML workloads on compute systems with distributed compute and communication across CPUs, GPUs, and network devices. The framework achieves highly accurate performance prediction without running workloads on target hardware.

**Key Contributions**:
- Geomean error of 5.21% for DLRM models on multi-GPU platforms
- Generalizes to Transformer-based NLP models with 3.00% error
- 85% success rate in selecting fastest embedding table sharding configuration
- Handles complexity of synchronization, load balancing, and various communication topologies

## Core Ideas

The prediction pipeline accurately predicts per-iteration training time across random configurations and generalizes well to different ML workload types. It addresses challenges including CPU-GPU synchronization, data distribution variance, and diverse communication devices (NVLink, PCIe, network cards), enabling rapid configuration optimization without extensive hardware benchmarking.

---

*Reading date: 2024*
*Note status: Completed*
