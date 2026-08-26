# Safeguarding LLM Training at Scale: Online SDC Detection and Insights from 35 Million GPU Hours

<div class="paper-meta" markdown>

**Authors**: Kinman Lei, Liyan Zheng, Xiang Li, Hongmin Chen, Yun Zhang, et al.  
**Institution**: Tsinghua University; ByteDance  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/lei)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">sdc-detection</span>
<span class="paper-tag">online-verification</span>
<span class="paper-tag">fleet-study</span>
</div>

## Background

Offline diagnostics do not protect live execution, full reruns duplicate enormous compute, and simple numerical thresholds lack accuracy. A production detector must be cheap continuously but definitive when suspicious.

## Methodology

AEGIS separates lightweight **cSensors** from selective **cVerifiers**. Sensors exploit training invariants and GPU characteristics to flag corruption candidates; verifiers spend additional work only on candidates to confirm and localize incidents.

## Experiment

Over 35 million GPU-hours, AEGIS detects 18 real SDC incidents and 13 faulty GPUs with 0.86% training overhead, providing a fleet-scale empirical characterization.

## Limitation

- Sensor coverage defines which corruptions can trigger verification.
- Verification latency can allow some polluted steps/checkpoints.
- Results reflect one production fleet and training stack.

---

*Reading date: 2026-08*
*Note status: Completed*
