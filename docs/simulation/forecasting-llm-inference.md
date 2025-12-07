# Forecasting LLM Inference Performance via Hardware-Agnostic Analytical Modeling

<div class="paper-meta" markdown>

**Authors**: Rajeev Patwari, Ashish Sirasao, Devleena Das
**Institution**: AMD
**Conference**: arXiv 2025
**Paper Link**: [arXiv:2508.00904](https://arxiv.org/abs/2508.00904)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">LLM Inference</span>
<span class="paper-tag">Performance Forecasting</span>
<span class="paper-tag">Hardware-Agnostic</span>
</div>

## Abstract

LIFE (LLM Inference Forecast Engine) is a lightweight analytical framework for modeling LLM inference workloads in a hardware and data-agnostic manner. It predicts TTFT, TPOT and TPS using only TOPS and bandwidth, enabling hardware-aware performance estimation without requiring benchmarking datasets.

**Key Contributions**:
- Analytical models of core LLM operators
- Configurable analytical workload model supporting various datatypes and optimizations
- Empirical analysis of prefill/decode phases under varying prompt/generation lengths
- Support for quantization, KV cache compression, chunked prefill, and different attention mechanisms

## Core Ideas

LIFE provides hardware-agnostic performance forecasting that has been verified with actual inference of Llama2-7B on AMD Ryzen CPU, NPU, iGPU and NVIDIA V100 GPU hardware. The framework enables rapid performance estimation across different hardware platforms without extensive benchmarking.

---

*Reading date: 2024*
*Note status: Completed*
