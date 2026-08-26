# Sailor: Automating Distributed Training over Dynamic, Heterogeneous, and Geo-Distributed Clusters

<div class="paper-meta" markdown>

**Authors**: Foteini Strati, Zhendong Zhang, George Manos, Ixeia Sanchez Periz, Qinghao Hu, Tiancheng Chen, Berk Buzcu, Song Han, Pamela Delgado, Ana Klimovic  
**Institution**: ETH Zurich; MIT; HES-SO  
**Conference**: SOSP '25  
**Year**: 2025  
**Paper Link**: [arXiv:2504.17096](https://arxiv.org/abs/2504.17096)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">distributed-training</span>
<span class="paper-tag">heterogeneous-clusters</span>
<span class="paper-tag">planning</span>
</div>

## Background

Large homogeneous allocations are scarce; mixing GPU types and zones can improve availability and cost but creates stragglers, topology-dependent communication, and a combinatorial resource/parallelism search space.

## Methodology

Sailor combines profiling, memory/runtime simulation, heuristic pruning, and dynamic programming. It jointly selects GPU allocation, layer-to-stage placement, DP/TP/PP degrees, microbatching, and geo placement for a throughput/cost objective, then executes the plan in a heterogeneity-aware training runtime.

## Experiment

| Metric | Result |
| --- | --- |
| Throughput | 1.1x-2.87x over planners on homogeneous/heterogeneous resources |
| Geo-distributed cost | 5.9x lower than DTFM in the reported comparison |
| Cost-constrained optimization | 40% saving over the second-best baseline |
| Search | Under one second for the cited 128-A100 case |

## Limitation

- Simulator error can select the wrong plan under transient contention.
- Cross-region training may be economically unattractive despite feasible throughput.
- Reconfiguration and checkpoint state movement still interrupt dynamic jobs.

---

*Reading date: 2026-08*
*Note status: Completed*
