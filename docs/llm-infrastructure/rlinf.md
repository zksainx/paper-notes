# RLinf: Flexible and Efficient Large-Scale Reinforcement Learning via Macro-to-Micro Flow Transformation

<div class="paper-meta" markdown>

**Authors**: Chao Yu, Yuanqing Wang, Zhen Guo, Hao Lin, Si Xu, et al.  
**Institution**: Tsinghua; Infinigence AI; Peking University; UC Berkeley; other collaborators  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/yu-chao)  
**GitHub**: [RLinf/RLinf](https://github.com/RLinf/RLinf)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">reinforcement-learning</span>
<span class="paper-tag">workflow-runtime</span>
<span class="paper-tag">elastic-pipeline</span>
</div>

## Background

Reasoning and embodied RL combine generation, training, reward/critic models, tools, and simulators with incompatible scaling modes. A single co-located or disaggregated execution template cannot fit every workflow.

## Methodology

M2Flow decomposes a high-level RL program spatially and temporally, then recomposes micro-flows into a profiled execution plan. Adaptive worker communication, context switching, and elastic pipelines allow mixtures of colocation and disaggregation without rewriting the logical workflow.

## Experiment

Across reasoning and embodied RL, RLinf improves end-to-end training throughput by 1.07-2.43x over state-of-the-art systems.

## Limitation

- Profiling and flow transformation enlarge the runtime/control surface.
- Correctness depends on preserving workflow dependencies through transformations.
- Dynamic external environments may invalidate an offline plan.

---

*Reading date: 2026-08*
*Note status: Completed*
