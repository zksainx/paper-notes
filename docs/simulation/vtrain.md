# vTrain: A Simulation Framework for Evaluating Cost-Effective and Compute-Optimal Large Language Model Training

<div class="paper-meta" markdown>

**Authors**: Research Team
**Institution**: KAIST and collaborators
**Conference**: MICRO 2024
**Paper Link**: [arXiv:2312.12391](https://arxiv.org/abs/2312.12391)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">LLM Training</span>
<span class="paper-tag">Cost Optimization</span>
<span class="paper-tag">Training Simulation</span>
</div>

## Abstract

vTrain is a profiling-driven simulator providing a fast yet accurate software framework to determine efficient and cost-effective LLM training system configurations. It addresses the critical challenge of training large models cost-effectively by exploring the parallelization design space.

**Key Contributions**:
- Fast design space exploration (under 200 seconds for full exploration)
- Identifies competitive training plans within minutes
- Supports multi-tenant GPU cluster scheduling optimization
- Determines compute-optimal model architectures given fixed compute budget

## Core Ideas

vTrain addresses limitations of heuristic-based parallel training strategies that leave significant performance on the table, wasting millions of dollars in training costs. Through case studies, it demonstrates effectiveness in evaluating optimal parallelization strategies balancing training time and cost, efficient multi-tenant scheduling, and compute-optimal model architecture selection.

---

*Reading date: 2024*
*Note status: Completed*
