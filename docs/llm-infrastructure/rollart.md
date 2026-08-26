# RollArt: Disaggregated Multi-Task Agentic RL Training at Scale

<div class="paper-meta" markdown>

**Authors**: Wei Gao, Yuheng Zhao, Tianyuan Wu, Shaopan Xiong, Weixun Wang, et al.  
**Institution**: HKUST; Alibaba Group; Tongyi Lab  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/gao)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">agentic-rl</span>
<span class="paper-tag">heterogeneous-hardware</span>
<span class="paper-tag">trajectory-runtime</span>
</div>

## Background

Agentic RL mixes compute-bound prefill, bandwidth-bound decode, CPU-bound stateful environments, bursty stateless rewards, and interconnect-heavy training. Batch-level coupling lets one failed or slow environment block unrelated trajectories.

## Methodology

RollArt maps prefill/decode/training to best-fit GPUs, environments to CPU clusters, and reward to serverless workers. Trajectory-level decoupling lets generation, environment, and reward advance independently; bounded-staleness weight synchronization overlaps rollout with training.

## Experiment

| Result | Evidence |
| --- | --- |
| Training time | 1.31-2.05x reduction against evaluated RL systems |
| Straggler handling | Up to 1.62x rollout speedup in injected-straggler tests |
| Production | Hundreds-of-billions MoE training on more than 3,000 GPUs; 1.66x end-to-end speedup over first 25 steps |

## Limitation

- Staleness bound trades utilization against RL fidelity.
- Hardware affinity is task-dependent and currently profile-driven.
- Serverless environments/reward paths introduce external failure and tail latency.

---

*Reading date: 2026-08*
*Note status: Completed*
