# AMALI: An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs

<div class="paper-meta" markdown>

**Authors**: Shiheng Cao, Junmin Wu, Junshi Chen, Hong An, Zhibin Yu
**Institution**: Multiple Institutions
**Conference**: ISCA 2025
**Paper Link**: [ACM DL](https://dl.acm.org/doi/10.1145/3695053.3731064)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Performance Modeling</span>
<span class="paper-tag">LLM Inference</span>
<span class="paper-tag">GPU Architecture</span>
</div>

## Abstract

AMALI addresses the challenge of accurately modeling LLM inference workloads on modern GPUs. The model provides detailed performance prediction for LLM inference by incorporating specialized models for tensor cores, constant cache, and instruction cache.

**Key Contributions**:
- Instruction modifier and throughput-based tensor core model that captures math pipe throttle stalls
- Analytical models for constant cache and instruction cache developed using micro-benchmarks
- Multi-warp model leveraging warp instruction number distribution for LLM inference characteristics

## Core Ideas

AMALI reduces mean absolute percentage error (MAPE) from 127.56% to 23.59% compared to the state-of-the-art GCoM model when validated on an A100 GPU. The framework can also be used for architecture design space exploration, accurately predicting end-to-end performance improvements with enhanced tensor core capability on H100.

---

*Reading date: 2024*
*Note status: Completed*
