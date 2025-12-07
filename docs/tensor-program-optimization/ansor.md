# Ansor: Generating High-Performance Tensor Programs for Deep Learning

<div class="paper-meta" markdown>

**Authors**: Lianmin Zheng, Chengfan Jia, Minmin Sun, Zhao Wu, Cody Hao Yu, Ameer Haj-Ali, Yida Wang, Jun Yang, Danyang Zhuo, Koushik Sen, Joseph E. Gonzalez, Ion Stoica  
**Institution**: UC Berkeley, University of Washington, AWS  
**Conference**: OSDI 2020  
**Paper Link**: [arXiv:2006.06762](https://arxiv.org/abs/2006.06762)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Tensor Programs</span>
<span class="paper-tag">Auto-Scheduling</span>
<span class="paper-tag">TVM</span>
</div>

## Abstract

Ansor is a tensor program generation framework for deep learning that automatically generates high-performance code without manual templates. It explores a large search space by sampling from a hierarchical representation and fine-tunes programs with evolutionary search and learned cost models.

**Key Contributions**:
- Template-free automatic tensor program generation
- Hierarchical search space sampling with evolutionary search
- Learned cost model for program performance prediction
- Improves performance by up to 3.8× (Intel CPU), 2.6× (ARM CPU), 1.7× (NVIDIA GPU)

## Core Ideas

Unlike AutoTVM which requires manual templates, Ansor takes only tensor expressions as input and generates high-performance code automatically. It has three major components: a program sampler for diverse program generation, a performance tuner using evolutionary search, and a task scheduler for time resource allocation. Ansor is integrated into Apache TVM as the tvm.auto_scheduler package.

---

*Reading date: 2025*
*Note status: Completed*
