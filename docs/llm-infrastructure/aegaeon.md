# Aegaeon: Effective GPU Pooling for Concurrent LLM Serving on the Market

<div class="paper-meta" markdown>

**Authors**: Yuxing Xiang, Xue Li, Kun Qian, Yufan Yang, Diwen Zhu, Wenyuan Yu, Ennan Zhai, Xuanzhe Liu, Xin Jin, Jingren Zhou  
**Institution**: Peking University; Alibaba Group  
**Conference**: SOSP '25  
**Year**: 2025  
**Paper Link**: [Author PDF](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">multi-model-serving</span>
<span class="paper-tag">gpu-pooling</span>
<span class="paper-tag">autoscaling</span>
</div>

## Background

Model markets have a long tail of bursty, low-rate models. Dedicated instances waste GPUs, while serverless/model-switching systems usually sustain only two or three models per GPU and pay seconds of initialization latency.

## Methodology

Aegaeon schedules and auto-scales at token boundaries. It keeps model state across GPU/host tiers, predicts switching/generation latency, prefetches weights, transfers KV state for preempted requests, and makes per-token placement decisions under SLO slack.

## Experiment

| Metric | Result |
| --- | --- |
| Pool density | Up to seven models per GPU |
| Goodput | Up to 2.5x over ServerlessLLM |
| Autoscaling latency | Full-stack optimizations reduce it by up to 97% |
| Deployment | Beta deployment on production model-market workloads |

## Limitation

- Token-granular scheduling adds control overhead and frequent state movement.
- Tight SLOs reduce pooling opportunities.
- Model popularity prediction errors can trigger costly switches.

---

*Reading date: 2026-08*
*Note status: Completed*
