# DiffKV: Differentiated Memory Management for Large Language Models with Parallel KV Compaction

<div class="paper-meta" markdown>

**Authors**: Yanqi Zhang, Yuwei Hu, Runyuan Zhao, John C. S. Lui, Haibo Chen  
**Institution**: Huawei; CUHK; Shanghai Jiao Tong University  
**Conference**: SOSP '25  
**Year**: 2025  
**Paper Link**: [arXiv:2412.03131](https://arxiv.org/abs/2412.03131)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">kv-cache</span>
<span class="paper-tag">compression</span>
<span class="paper-tag">gpu-memory</span>
</div>

## Background

Uniform pruning or quantization ignores that keys and values, heads, layers, tokens, and requests have different sensitivity. Differentiated compression improves quality per byte but creates tens of thousands of irregular memory regions that a CPU allocator cannot update every decode step.

## Methodology

| Data structure | Purpose |
| --- | --- |
| Unified pages | Encode different K/V precision configurations in fixed-size GPU pages |
| Circular free-page list | Keep free/used regions compact and support parallel allocation |
| Bidirectional page table | Grow high- and low-precision mappings from opposite ends |
| Parallel compaction | Plan per-head needs and coordinate movement with GPU prefix sums |

The memory manager launches across `(sequence, layer, KV head)`, keeping allocation and recycling on GPU. This makes differentiated precision practical without serial control-plane overhead.

## Experiment

| Metric | Result |
| --- | --- |
| KV compression | 2.7x-5.7x with near-lossless accuracy |
| Serving throughput | 1.9x-5.4x improvement |
| Models | Includes QwQ-32B and DeepSeek-R1 distilled 14B/8B reasoning models |

## Limitation

- Compression policy and accuracy sensitivity remain model/task dependent.
- GPU compaction consumes compute and memory bandwidth that may contend with inference.
- More precision tiers increase page-table and kernel complexity.

---

*Reading date: 2026-08*
*Note status: Completed*
