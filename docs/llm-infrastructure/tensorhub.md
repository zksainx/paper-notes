# TensorHub: Scalable and Elastic Weight Transfer for LLM RL Training

<div class="paper-meta" markdown>

**Authors**: Chenhao Ye, Huaizheng Zhang, Mingcong Han, Baoquan Zhong, Xiang Li, Qixiang Chen, et al.  
**Institution**: University of Wisconsin-Madison; ByteDance Seed  
**Conference**: SOSP '26  
**Year**: 2026  
**Paper Link**: [arXiv:2604.09107](https://arxiv.org/abs/2604.09107)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">weight-transfer</span>
<span class="paper-tag">rl-training</span>
<span class="paper-tag">reference-storage</span>
</div>

## Background

NCCL is fast but assumes static groups; P2P is elastic but fan-out contends; object/parameter storage decouples membership but doubles TB-scale movement through push-then-pull and owns extra copies.

## Methodology

Reference-Oriented Storage tracks immutable weight versions already replicated on rollout GPUs and exposes them as storage objects without copying. A mutability contract, retention protocol, topology-aware peer selection, pipelined replication, model-parallel consistency, and failure masking turn this abstraction into TensorHub.

## Experiment

TensorHub transfers 50 GB at 22 GB/s (88% theoretical RDMA). It reduces total GPU stall by up to 6.7x on 1,024 GPUs, accelerates elastic spot-worker updates 4.8x versus UCX, and cuts cross-DC stall 19x by paying one seed transfer then using local RDMA. It is deployed in ByteDance RL training.

## Limitation

- A retained version can become temporarily unavailable if its last reliable replica fails.
- Correctness depends on immutable version/retention contracts.
- Cross-DC seeding still requires one full WAN transfer and optional CPU copy.

---

*Reading date: 2026-08*
*Note status: Completed*
