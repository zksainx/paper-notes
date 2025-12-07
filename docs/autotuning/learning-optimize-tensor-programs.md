# Learning to Optimize Tensor Programs

<div class="paper-meta" markdown>

**Authors**: Tianqi Chen, Lianmin Zheng, Eddie Q. Yan, Ziheng Jiang, Thierry Moreau, et al.
**Institution**: University of Washington, AWS, Cornell
**Conference**: NeurIPS 2018
**Paper Link**: [arXiv:1805.08166](https://arxiv.org/abs/1805.08166)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">AutoTVM</span>
<span class="paper-tag">Machine Learning</span>
<span class="paper-tag">Compiler Optimization</span>
</div>

## Abstract

This paper introduces AutoTVM, a learning-based framework to optimize tensor programs for deep learning workloads. It learns domain-specific statistical cost models to guide the search of tensor operator implementations over billions of possible program variants.

**Key Contributions**:
- Machine learning-based tensor program optimization framework
- Statistical cost models for guiding program search
- Transfer learning acceleration (2-10× speedup)
- End-to-end performance improvements of 1.2× to 3.8×

## Core Ideas

AutoTVM addresses the limitation of manually optimized libraries like cuDNN that only support narrow ranges of hardware. It generates programs competitive with hardware-specific libraries while enabling operator fusion optimizations impossible with fixed operator libraries. The framework formalizes the optimization problem and proposes an ML solution that generalizes across different hardware targets.

---

*Reading date: 2024*
*Note status: Completed*
