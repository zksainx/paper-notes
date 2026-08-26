# DynaRL: Flexible and Dynamic Scheduling of Large-Scale Reinforcement Learning Training

<div class="paper-meta" markdown>

**Authors**: Yuanqing Wang, Hao Lin, Junhao Hu, Chunyang Zhu, Quanlu Zhang, et al.  
**Institution**: Peking University; Infinigence AI; Tsinghua; ICT CAS; other collaborators  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">dynamic-scheduling</span>
<span class="paper-tag">agentic-rl</span>
<span class="paper-tag">resource-migration</span>
</div>

## Background

Heavy-tailed rollout lengths, multi-turn tools, and time-varying bottlenecks can waste up to 60% of compute. Static GPU partitions cannot follow the bottleneck as it moves among rollout, tools, reward, and training.

## Methodology

DynaRL represents the live pipeline as a dynamic hypergraph. A multi-level scheduler identifies the bottleneck and migrates compute, memory, and communication resources through a unified interface; context-aware routing preserves state/KV locality during movement.

## Experiment

DynaRL improves end-to-end throughput by up to 1.98x on math-reasoning and agentic RL workloads with negligible online scheduling overhead.

## Limitation

- Migration can be expensive for large model/KV/optimizer state.
- Centralized hypergraph control is a scalability and reliability concern.
- Very fast demand oscillations can outrun the control loop.

---

*Reading date: 2026-08*
*Note status: Completed*
