# DirectKV: Efficient Zero-Copy KV Cache Offloading for Long-Context LLMs

<div class="paper-meta" markdown>

**Authors**: Shutian Luo, Haiying Shen  
**Institution**: University of Virginia  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/luo)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">zero-copy</span>
<span class="paper-tag">kv-cache</span>
<span class="paper-tag">grace-hopper</span>
</div>

## Background

Conventional KV offload stages CPU-resident blocks back into HBM, doubling transfers and reserving scarce GPU buffers. NVLink-C2C superchips make direct GPU access to CPU memory plausible.

## Methodology

DirectKV removes the staging buffer, introduces CPU-memory-aware attention kernels, fuses KV generation with attention, and pipelines fetch/compute/write-back at warp level.

## Experiment

| Metric on GH200 | Result |
| --- | --- |
| Transfer volume | Up to 50% reduction |
| GPU memory | 43% reduction |
| End-to-end | Up to 1.2x improvement |
| Kernel effect | Up to 70% lower inference latency in an ablation |

## Limitation

- The throughput case relies on NVLink-C2C bandwidth; PCIe is usually too slow.
- Direct remote loads still expose CPU-memory latency.
- Specialized attention kernels reduce portability.

---

*Reading date: 2026-08*
*Note status: Completed*
