# PrismLLM: Faithful Large-Scale LLM Training Emulation with a Few GPUs

<div class="paper-meta" markdown>

**Authors**: Shaoke Xi, ChonLam Lao, Boyi Jia, Jiaqi Gao, Zhipeng Zhang, Jiamin Cao, et al.  
**Institution**: Alibaba Group; SJTU; Harvard; Zhejiang University  
**Conference**: SOSP '26  
**Year**: 2026  
**Paper Link**: [arXiv:2605.15617](https://arxiv.org/abs/2605.15617)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">training-emulation</span>
<span class="paper-tag">execution-slicing</span>
<span class="paper-tag">performance-debugging</span>
</div>

## Background

Simulation needs continuously maintained performance models, while downscaled real runs change parallelism, communication dependencies, memory layout, and bottlenecks. Engineers need production-scale behavior without reserving production-scale hardware.

## Methodology

PrismLLM slices a target execution graph around ranks of interest. **Sandbox ranks** execute the real framework/kernels; **virtual participants** replay omitted ranks through assistant nodes. Virtualized initialization, collective communication pruning, mock routing, and inter-slice timing calibration preserve dependencies/control flow while avoiding irrelevant work.

| Mechanism | Purpose |
| --- | --- |
| Graph slicing | Retain only dependency paths influencing observed ranks |
| Hybrid emulation | Run selected ranks physically and replay peers virtually |
| Communication pruning | Complete irrelevant collectives without transferring payloads |
| Calibration | Align slices to target-scale timing and contention |

## Experiment

Iteration-time error averages 0.58% (1.98% maximum) and peak-memory error stays below 0.01%, including reproducing OOMs. An 8,192-GPU target uses 32 physical nodes (<0.4% of target GPUs), completes emulation within 80 minutes, and bootstraps in 44.3 s using 5.4 GB/device.

## Limitation

- Control-dependent tensors require context switching or manually supplied generation rules.
- Stored rank contexts can reach TB scale and spill to disk.
- Resource contention among omitted ranks is approximated rather than physically reproduced.

---

*Reading date: 2026-08*
*Note status: Completed*
