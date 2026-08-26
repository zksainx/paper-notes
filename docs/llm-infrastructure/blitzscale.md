# BlitzScale: Fast and Live Large Model Autoscaling with O(1) Host Caching

<div class="paper-meta" markdown>

**Authors**: Dingyan Zhang, Haotian Wang, Yang Liu, Xingda Wei, Yizhou Shan, Rong Chen, Haibo Chen  
**Institution**: Shanghai Jiao Tong University; Huawei Cloud  
**Conference**: OSDI '25  
**Year**: 2025  
**Paper Link**: [USENIX paper](https://www.usenix.org/conference/osdi25/presentation/zhang-dingyan)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">autoscaling</span>
<span class="paper-tag">llm-serving</span>
<span class="paper-tag">gpu-networking</span>
</div>

## Background

Model-as-a-service platforms need to absorb second-scale request bursts without permanently reserving GPUs. Existing autoscalers load 10-400 GB model checkpoints from SSD or host DRAM and keep a new instance stopped until all parameters arrive. Large host caches improve hit latency but do not scale across many models; ServerlessLLM reports only 40-75% cache hit rates in such settings.

**Key Takeaways**

- BlitzScale moves parameter loading to the underutilized GPU compute fabric and uses multicast so multiple replicas do not each require a full host cache.
- It makes scaling live by decomposing an instance into layer-level work: a new GPU can execute loaded layers before the complete model arrives.
- The system reframes autoscaling as a data-plane and execution-scheduling problem rather than a cache-size problem.

## Methodology

| Mechanism | Design |
| --- | --- |
| Compute-network loading | Transfers parameters over GPU-GPU/CPU-GPU fabrics instead of relying on SSD reads |
| Network multicast | One source instance streams a parameter layer to multiple scaled replicas |
| Layer-level live scaling | Existing instances offload ready layers to a partially loaded replica |
| Scheduler | Coordinates layer availability, request routing, and pipeline execution |

The key constraint is preserving correctness while a replica is only partially initialized. BlitzScale therefore exposes fine-grained layer readiness to the serving scheduler instead of presenting a binary “instance started/stopped” abstraction.

## Experiment

| Metric | Result |
| --- | --- |
| Tail latency | Up to 94% lower than ServerlessLLM on real-world workloads |
| GPU time | 49% less than DistServe/vLLM under the same SLA |
| Motivation example | Loading Llama3-8B through a 10 Gbps SSD takes 12.8 s, far beyond sub-second chatbot SLOs |

The paper evaluates bursty multi-model workloads and compares cache-based and non-autoscaling baselines. The improvement comes from both faster data movement and the ability to serve during loading; either optimization alone leaves a stop-the-world gap.

## Limitation

- The approach assumes a high-bandwidth compute network and requires serving-engine changes.
- Multicast traffic can contend with inference traffic and must be scheduled carefully.
- Layer-level execution is less useful for models with irregular control flow or highly coupled state.

---

*Reading date: 2026-08*
*Note status: Completed*
