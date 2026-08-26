# Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference

<div class="paper-meta" markdown>

**Authors**: Le Chen, Dahu Feng, Erhu Feng, Yingrui Wang, Rong Zhao, Yubin Xia, Pinjie Xu, Haibo Chen  
**Institution**: Shanghai Jiao Tong University; Tsinghua University; SenseTime Research  
**Conference**: SOSP '25  
**Year**: 2025  
**Paper Link**: [arXiv:2501.14794](https://arxiv.org/abs/2501.14794)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">mobile-soc</span>
<span class="paper-tag">gpu-npu</span>
<span class="paper-tag">on-device-inference</span>
</div>

## Background

Mobile SoCs share memory across CPU, GPU, and NPU, yet engines usually select one accelerator. NPU performance is highly sensitive to tensor shape and operand order, and no single processor saturates total SoC memory bandwidth.

## Methodology

HeteroInfer profiles operator affinity and partitions tensors across GPU/NPU for concurrent execution. It uses unified-memory synchronization to avoid redundant copies and selects weight- or activation-dimension partitioning according to prefill/decode shape.

## Experiment

| Metric | Result |
| --- | --- |
| End-to-end | 1.34x-6.02x faster than GPU-only/NPU-only engines |
| Prefill comparison | Up to 3.69x over a reported NPU baseline |
| Padding/shape optimization | Up to 2.12x improvement |
| Co-running apps | Negligible reported interference, including gaming cases |

## Limitation

- NPU APIs and performance behavior are vendor/device specific.
- Shared memory removes copies but not bandwidth contention.
- Partitioning requires accurate per-shape profiling.

---

*Reading date: 2026-08*
*Note status: Completed*
