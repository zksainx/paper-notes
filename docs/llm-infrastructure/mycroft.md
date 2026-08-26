# Mycroft: Tracing Dependencies in Collective Communication Towards Reliable LLM Training

<div class="paper-meta" markdown>

**Authors**: Yangtao Deng, Lei Zhang, Qinlong Wang, Xiaoyun Zhi, Xinlei Zhang, Zhuo Jiang, et al.  
**Institution**: CUHK; ByteDance; ByteDance Seed; Harvard University  
**Conference**: SOSP '25  
**Year**: 2025  
**Paper Link**: [arXiv:2509.03018](https://arxiv.org/abs/2509.03018)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">collectives</span>
<span class="paper-tag">root-cause-analysis</span>
<span class="paper-tag">observability</span>
</div>

## Background

Collective communication libraries expose coarse API timing while hiding flow/chunk progress and dependencies. A slow rank can propagate delay through ring/tree collectives and across overlapping DP/PP groups, making many innocent ranks look faulty. Full kernel tracing produces too much data for continuous production use.

## Methodology

| Layer | Signal | Diagnostic value |
| --- | --- | --- |
| Collective group | Operation duration/progress | Finds affected DP/TP/PP groups |
| Flow | Rank-to-rank dependencies | Separates origin from propagated slowdown |
| Chunk | Aggregated transfer progress | Attributes GPU, CPU, RDMA, or software stalls |
| Trigger | Sampled anomaly state | Activates heavier analysis only when needed |

Mycroft instruments NCCL internal streams, writes into a fixed shared-memory circular buffer, and uploads asynchronously through a read-only agent. Dependency-driven analysis traces blocked progress backward to the earliest violating component.

## Experiment

| Metric | Result |
| --- | --- |
| Production deployment | More than six months at ByteDance |
| Detection | 90% of anomalies within 15 s; 100% of collective-level cases detected |
| Root cause | 60% within 20 s; all evaluated cases within one minute |
| Trace volume | About 46.8 KB per iteration per machine; about 186 KB at 1,200-machine scale |
| Overhead | Near baseline on 32 A100 NCCL tests; NPKit reduces bandwidth to roughly one third in the comparison |

## Limitation

- Mycroft diagnoses dependencies inside collectives; it needs integration with stack, hardware, or application tools for final repair.
- Database retrieval dominates analysis time as scale grows.
- Instrumentation relies on access to collective-library internals.

---

*Reading date: 2026-08*
*Note status: Completed*
