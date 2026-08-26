# NanoFlow: Towards Optimal Large Language Model Serving Throughput

<div class="paper-meta" markdown>

**Authors**: Kan Zhu, Yufei Gao, Yilong Zhao, Liangyu Zhao, Gefei Zuo, Yile Gu, Dedong Xie, Tian Tang, Qinyu Xu, Zihao Ye, Keisuke Kamahori, Chien-Yu Lin, Ziren Wang, Stephanie Wang, Arvind Krishnamurthy, Baris Kasikci  
**Institution**: University of Washington; Tsinghua University; UC Berkeley; University of Michigan  
**Conference**: OSDI '25  
**Year**: 2025  
**Paper Link**: [USENIX paper](https://www.usenix.org/conference/osdi25/presentation/zhu-kan)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">llm-serving</span>
<span class="paper-tag">pipeline-scheduling</span>
<span class="paper-tag">throughput</span>
</div>

## Background

LLM serving is often labeled memory-bound because every decode step loads weights and KV state. NanoFlow's profiling shows that end-to-end execution is frequently compute-bound instead: compute, memory, and network operations are serialized inside a device, leaving one resource idle while another is active. The target is throughput, measured as tokens per device per second, under realistic concurrent workloads.

**Key Takeaways**

- NanoFlow creates intra-device parallelism by splitting requests into nano-batches and duplicating operators.
- A search procedure chooses nano-batch count, size, order, and resource allocation while modeling interference among concurrent operations.
- The system raises utilization without requiring a new accelerator or changing the model.

## Methodology

| Component | Function |
| --- | --- |
| Nano-batching | Splits an input batch into independently schedulable pieces |
| Operator duplication | Creates parallel copies so compute/memory/network phases can overlap |
| Resource-aware planner | Searches batch sizes, order, and GPU resource assignments |
| Interference model | Accounts for contention between concurrent kernels and transfers |

The design is deliberately finer-grained than layer pipelining. It exposes enough independent work within a device to overlap heterogeneous resources, but keeps the execution schedule model-aware so extra duplication does not erase the gain.

## Experiment

| Setting | Result |
| --- | --- |
| Models | LLaMA-2-70B, Mixtral 8x7B, LLaMA-3-8B and other popular models |
| Practical workloads | 1.91x throughput improvement over state-of-the-art serving systems |
| Distance to optimum | 50-72% of an estimated optimal throughput across popular models |

The paper's “optimal” reference is a resource-overlap upper bound rather than an unattainable oracle implementation. The remaining gap comes from model dependencies, kernel granularity, and the cost of duplicating/launching operations.

## Limitation

- NanoFlow increases scheduling and implementation complexity inside the serving engine.
- The planner must be retuned when kernels, GPU generations, or model shapes change.
- Throughput gains can trade against per-request latency and memory footprint.

---

*Reading date: 2026-08*
*Note status: Completed*
