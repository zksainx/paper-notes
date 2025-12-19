# The Anatomy of a Triton Attention Kernel

<div class="paper-meta" markdown>

**Authors**: Burkhard Ringlein, Jan van Lunteren, Radu Stoica  
**Institution**: IBM Research Europe  
**Conference**: arXiv 2025  
**Paper Link**: [arXiv:2511.11581](https://arxiv.org/abs/2511.11581)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Triton</span>
<span class="paper-tag">Attention Mechanism</span>
<span class="paper-tag">GPU Kernels</span>
</div>

## Abstract

This paper develops a state-of-the-art paged attention kernel using exclusively OpenAI's Triton language, achieving state-of-the-art performance on both NVIDIA and AMD GPUs. It brings Triton attention kernel performance from 19.7% to 105.9% of state-of-the-art through systematic optimizations.

**Key Contributions**:
- Feature-complete cross-platform paged attention kernel
- High-level approach with algorithmic and system-level improvements
- Parameter auto-tuning for optimal performance
- Open-sourced kernels adopted as default in vLLM for AMD GPUs

## Kernel Parameter Autotuning

通过运行微基准测试（Microbenchmarks）进行“经验性试错”来锁定最佳内核参数(BLOCK_SIZE, num_stages)，从而在大幅缩减编译器所需处理的参数搜索空间的同时，以极低的人工成本实现了远超纯编译器静态分析的优化深度与跨硬件性能可移植性 。


## Implementation Details






---

*Reading date: 2025*
*Note status: Completed*
