# A Learned Performance Model for Tensor Processing Units

<div class="paper-meta" markdown>

**Authors**: Samuel J. Kaufman, Phitchaya Mangpo Phothilimthana, Yanqi Zhou  
**Institution**: Google  
**Conference**: MLSys 2021  
**Paper Link**: [arXiv:2008.01040](https://arxiv.org/abs/2008.01040)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">TPU</span>
<span class="paper-tag">Performance Modeling</span>
<span class="paper-tag">Machine Learning</span>
</div>

## Abstract

This paper demonstrates a method of learning performance models from a corpus of tensor computation graph programs for Tensor Processing Unit (TPU) instances. The learned model outperforms heavily-optimized analytical performance models on tile-size selection and operator fusion tasks.

**Key Contributions**:
- Machine learning-based performance modeling for TPUs
- Outperforms analytical models on key optimization tasks
- Enables autotuning in settings where TPU access is limited or expensive
- Reduces development burden for modeling contemporary accelerators

## Core Ideas

Accurate hardware performance models are critical for efficient code generation but difficult to develop for complex processors. This research shows that learning from execution data can produce more accurate models than traditional analytical approaches, particularly useful for the proliferation of deep learning accelerators where analytical modeling is costly.

---

*Reading date: 2025*
*Note status: Completed*
