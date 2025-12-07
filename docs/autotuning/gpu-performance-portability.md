# GPU Performance Portability needs Autotuning

<div class="paper-meta" markdown>

**Authors**: Burkhard Ringlein, Thomas Parnell, Radu Stoica  
**Institution**: IBM Research Europe  
**Conference**: arXiv 2025  
**Paper Link**: [arXiv:2505.03780](https://arxiv.org/abs/2505.03780)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">GPU Portability</span>
<span class="paper-tag">Autotuning</span>
<span class="paper-tag">LLM Inference</span>
</div>

## Abstract

This paper makes the case for combining just-in-time (JIT) compilation with comprehensive kernel parameter autotuning to enable portable LLM inference with state-of-the-art performance without code changes, addressing vendor lock-in and hardware portability challenges.

**Key Contributions**:
- Explores up to 15× more kernel parameter configurations
- Produces significantly more diverse code across multiple dimensions
- Outperforms vendor-optimized implementations by up to 230%
- Reduces kernel code size by 70× while eliminating manual optimizations

## Core Ideas

As LLMs grow in complexity, reliance on a single dominant platform limits portability and creates vendor lock-in. Combining JIT compilation with comprehensive autotuning enables performance-portable LLM inference across different GPU vendors. The "dejavu" mechanism for Triton's autotuner reduces overhead to zero, enabling production use with full autotuning benefits.

---

*Reading date: 2025*
*Note status: Completed*
