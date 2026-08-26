# Robust LLM Training Infrastructure at ByteDance

<div class="paper-meta" markdown>

**Authors**: Borui Wan, Gaohong Liu, Zuquan Song, Jun Wang, Yun Zhang, Guangming Sheng, et al.  
**Institution**: The University of Hong Kong; ByteDance Seed  
**Conference**: SOSP '25  
**Year**: 2025  
**Paper Link**: [arXiv:2509.16293](https://arxiv.org/abs/2509.16293)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">fault-tolerance</span>
<span class="paper-tag">llm-training</span>
<span class="paper-tag">production</span>
</div>

## Background

At 10K-GPU scale, failures are routine and a fail-stop/diagnose/reschedule/checkpoint-reload cycle can cost more than ten minutes. ByteRobust optimizes effective training time ratio (ETTR), not only mean time to recovery. Its operational insight is to resume training after rapidly isolating a small suspect set instead of blocking on perfect root-cause localization.

**Key Takeaways**

- A hierarchical control plane combines online detection, reattempt/rollback, eviction, and controlled group testing.
- Parallelism-aware group isolation supports faults where a precise GPU is not immediately identifiable.
- Warm backups, aggregated hot updates, and fault-aware every-step checkpointing reduce restart cost.

## Methodology

| Component | Role |
| --- | --- |
| Real-time checks | Detect explicit errors, hangs, NaNs, and health anomalies |
| Hierarchical diagnosis | Escalate from cheap tests to group testing only when necessary |
| Robust controller | Evict suspect machines, retry, roll back, or trigger failover |
| Fault-aware checkpoint | Preserve every-step recoverability with low overhead |
| Hot update / warm backup | Change code and recover capacity without full cluster rescheduling |

Across 19 jobs using at least 9,600 GPUs, direct eviction resolved 32.52% of incidents, reattempts 22.70%, and rollback 9.20%. This validates the “fast demarcation first” policy: many incidents do not require an exact physical root cause before productive work resumes.

## Experiment

| Setting | Result |
| --- | --- |
| Long production jobs | Up to 97% cumulative ETTR over three months on 9,600 GPUs |
| Checkpointing | Every-step checkpoint with <0.9% overhead |
| Failure recovery | Unproductive time kept within a maximum of about 50 minutes for reported dense/MoE runs |
| Scale test | 1,024 machines, 16 L20 GPUs each (16,384 GPUs total) for fault-handling evaluation |

## Limitation

- The system deliberately over-evicts healthy machines in some PP groups; this is economical only at very large scale.
- Diagnosis policies and failure distributions are specific to ByteDance's platform.
- ETTR can conceal model-quality damage from silent errors unless paired with correctness checks.

---

*Reading date: 2026-08*
*Note status: Completed*
