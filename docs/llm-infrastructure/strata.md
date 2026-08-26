# Strata: Hierarchical Context Caching for Long Context Language Model Serving

<div class="paper-meta" markdown>

**Authors**: Zhiqiang Xie, Ziyi Xu, Mark Zhao, Yuwei An, Vikram Sharma Mailthody, Scott Mahlke, Michael Garland, Christos Kozyrakis  
**Institution**: Stanford; NVIDIA; SJTU; CU Boulder; CMU; University of Michigan  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">context-cache</span>
<span class="paper-tag">gpu-io</span>
<span class="paper-tag">long-context</span>
</div>

## Background

Hierarchical KV caching extends capacity into CPU/SSD, but fragmented GPU pages produce small transfers, cache loads stall prefill, and concurrent requests for the same missing prefix create “delay hits.” In one SGLang setup, KV transfer blocks 74% of prefill time and cuts throughput by up to 4x.

## Methodology

Strata decouples host and GPU cache layouts: GPU-assisted I/O gathers/scatters pages while the host performs large transfers. Its scheduler accounts for cache-load time, batches complementary compute/I/O work, hides stalls, and coordinates concurrent misses.

## Experiment

| Metric | Result |
| --- | --- |
| Throughput | Up to 5x over vLLM-LMCache and 3.75x over TensorRT-LLM |
| Page-first host layout | 2.1x lower average TTFT and 1.3x higher throughput in one ablation |
| Deployment | Integrated with SGLang and deployed in production |

## Limitation

- GPU-assisted movement consumes kernels and memory bandwidth.
- Policies focus on transfer latency rather than deeper storage prefetch.
- Benefit depends on prefix reuse and cache-distance distribution.

---

*Reading date: 2026-08*
*Note status: Completed*
