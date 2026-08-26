# Tessera: Holistic Pipeline Parallelism for Trillion-Parameter Heterogeneous MoE Training

<div class="paper-meta" markdown>

**Authors**: Weifang Hu, Langshi Chen, Man Yuan, Youyang Yao, Xiulong Yuan, Li Tian, Yong Li, Wei Lin, Xuanhua Shi, Zhengping Qian, Jingren Zhou  
**Institution**: HUST; Alibaba Cloud  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/hu-weifang)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">moe-training</span>
<span class="paper-tag">pipeline-parallelism</span>
<span class="paper-tag">communication-overlap</span>
</div>

## Background

Qwen3-Next interleaves standard attention, Gated DeltaNet, and sparse MoE blocks. Equal serial layer cost no longer means equal pipeline cost because layer combinations hide different fractions of All-to-All communication; MoE routing also creates transient bubbles.

## Methodology

| Component | Role |
| --- | --- |
| Overlap scheduler | Synthesizes fine-grained compute/A2A interleavings per layer combination |
| Overlap-aware partitioner | Balances profiled post-overlap cost rather than serial cost |
| Dynamic bubble optimizer | Moves eligible tasks into routing-induced idle slots at runtime |

## Experiment

Across Qwen3/Qwen3-Next runs on 4,096-12,288 GPUs, Tessera raises throughput by 20-33%, reaches 39% MFU for a trillion-parameter model, and achieves up to 1.24x higher MFU than Megatron-Core MoE. Planning solves within five seconds after profiling, though one 128K profiling case takes about 3,050 seconds.

## Limitation

- Hardware profiling is expensive and tied to model/cluster configuration.
- Movable-task constraints limit dynamic bubble filling.
- Routing distributions can drift after the offline partition is chosen.

---

*Reading date: 2026-08*
*Note status: Completed*
