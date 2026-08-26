# KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models

<div class="paper-meta" markdown>

**Authors**: Hongtao Chen, Weiyu Xie, Boxin Zhang, Jingqi Tang, Jiahao Wang, et al.  
**Institution**: Tsinghua University; Approaching.AI; Hangzhou Dianzi University; other collaborators  
**Conference**: SOSP '25  
**Year**: 2025  
**Paper Link**: [Official publication page](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/)  
**GitHub**: [kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">moe-inference</span>
<span class="paper-tag">cpu-gpu-hybrid</span>
<span class="paper-tag">expert-deferral</span>
</div>

## Background

Sparse MoE models fit naturally in CPU DRAM with selected experts computed per token, but existing hybrid engines underuse modern CPU matrix units and serialize CPU/GPU tasks. This blocks local serving of models such as DeepSeek-V3/R1 671B despite their sparse activation.

## Methodology

| Component | Role |
| --- | --- |
| AMX-specialized kernels | Improve CPU expert GEMM layout, cache behavior, and instruction utilization |
| Async task scheduler | Overlap GPU attention/shared experts with CPU routed experts |
| Expert deferral | Delay selected expert work to enlarge overlap windows rather than dropping it outright |
| Model mapping | Assign dense latency-sensitive operators to GPU and sparse capacity-heavy experts to CPU |

## Experiment

| Phase | Result over existing hybrid systems |
| --- | --- |
| Prefill | 4.62x-19.74x speedup |
| Decode | 1.25x-4.09x speedup |
| Expert deferral | Up to 1.45x additional throughput; CPU utilization rises from below 75% to nearly 100% |
| Quality | No more than 0.5% average accuracy drop in the evaluated benchmarks |

## Limitation

- Performance depends heavily on AMX-capable CPUs and DRAM bandwidth.
- Expert deferral is approximate and its quality effect may vary by task/model.
- Low-concurrency local inference is the sweet spot; cloud-scale batching changes the optimal placement.

---

*Reading date: 2026-08*
*Note status: Completed*
