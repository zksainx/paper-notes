# Janus: Multi-LLM Serving at Production Scale

<div class="paper-meta" markdown>

**Authors**: Tianbao Zhou, Yi Wang, Yu Zhou, Zirui Liu, Zhiming Wang, Yebo Peng, et al.  
**Institution**: Peking University; JD; UCAS; USTC; SJTU  
**Conference**: SOSP '26 (camera-ready not yet public)  
**Year**: 2026  
**Paper Link**: [Official artifact](https://github.com/MachineLearningSystem/26SOSP-Janus)  
**GitHub**: [MachineLearningSystem/26SOSP-Janus](https://github.com/MachineLearningSystem/26SOSP-Janus)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">multi-llm-serving</span>
<span class="paper-tag">autoscaling</span>
<span class="paper-tag">production</span>
</div>

## Background

Production MaaS combines power-law model popularity, sub-second bursts, and complementary HBM/compute/bandwidth demands. A single sharing timescale either wastes steady tail capacity or reacts too slowly to head-model bursts.

## Methodology

Janus separates slow and fast control. Every 30 s, Gaussian-process oracles and vector bin packing colocate complementary tail models in a steady pool. Every 0.5 s, SLO feedback scales head models in an elastic pool. LST-IMH performs deadline routing with a 2-approximation guarantee.

At the engine, xTensor creates a virtual HBM page pool with weights growing upward and KV/activations downward. Active/warm/cold lifecycle states enable host-to-device wakeup or 388-783 ms device-to-device weight forks.

## Experiment

On a 768-device production cluster, 62 applications, and 45.6M requests, Janus maintains 0.97-1.0 SLO attainment while saving 24% device-hours versus the best baseline. Daily device-hours are 13,440 versus 17,856 for ServerlessLLM.

## Limitation

- Full results require a large Ascend CloudMatrix environment.
- GP models and dual timescales require retraining/tuning under workload drift.
- This note is based on the official artifact README because the camera-ready paper is not public yet.

---

*Reading date: 2026-08*
*Note status: Completed from official artifact; paper pending*
