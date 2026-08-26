# Jenga: Effective Memory Management for Serving LLM with Heterogeneity

<div class="paper-meta" markdown>

**Authors**: Chen Zhang, Kuntai Du, Shu Liu, Woosuk Kwon, Xiangxi Mo, Yufeng Wang, et al.  
**Institution**: Tsinghua University; University of Chicago; UC Berkeley  
**Conference**: SOSP '25  
**Year**: 2025  
**Paper Link**: [arXiv:2503.18292](https://arxiv.org/abs/2503.18292)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">memory-allocation</span>
<span class="paper-tag">heterogeneous-models</span>
<span class="paper-tag">prefix-cache</span>
</div>

## Background

PagedAttention assumes largely uniform KV embeddings and policies. Multimodal, hybrid-attention, and state-space models have different embedding widths and token dependencies across layer types, causing fragmentation and making one eviction policy incorrect for part of the model.

## Methodology

Jenga uses a two-level allocator. Large pages use the least common multiple of all embedding page sizes; type-specific allocators subdivide them into small pages. When every small page becomes free, the large page returns to the global allocator. APIs attach custom cache/eviction logic to each layer type.

| Choice | Benefit | Cost |
| --- | --- | --- |
| LCM large pages | Compatible with all embedding sizes without new attention kernels | Possible internal fragmentation |
| Request-aware small pages | Packs same-request allocations and recovers whole large pages | More allocator metadata |
| Type-specific eviction | Matches self/cross/hybrid attention dependencies | Policy complexity |

## Experiment

| Platform | Result |
| --- | --- |
| H100 80GB | Up to 4.92x throughput, 1.80x average |
| L4 24GB | Up to 3.29x throughput, 1.69x average |
| Memory utilization | Up to 79.6% improvement |

Jenga remains comparable to vLLM on standard Llama, showing that generalized heterogeneity support need not penalize uniform models.

## Limitation

- An extreme LCM can create large allocation units, though evaluated models remained manageable.
- Application-defined evictors can violate quality or reuse expectations.
- Results depend on the model mix and request-level prefix locality.

---

*Reading date: 2026-08*
*Note status: Completed*
