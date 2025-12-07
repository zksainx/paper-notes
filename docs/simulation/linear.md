# Linformer: Self-Attention with Linear Complexity

<div class="paper-meta" markdown>

**Authors**: Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma
**Institution**: Facebook AI
**Conference**: arXiv 2020
**Paper Link**: [arXiv:2006.04768](https://arxiv.org/abs/2006.04768)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Transformer</span>
<span class="paper-tag">Linear Attention</span>
<span class="paper-tag">Complexity Reduction</span>
</div>

## Abstract

Linformer reduces the overall self-attention complexity from O(n²) to O(n) in both time and space by demonstrating that the self-attention mechanism can be approximated by a low-rank matrix, making it practical for long sequence processing.

**Key Contributions**:
- Low-rank approximation of self-attention reducing complexity to linear
- Maintains competitive performance with standard Transformer
- Enables processing of much longer sequences efficiently

## Core Ideas

The standard self-attention mechanism has O(n²) complexity with respect to sequence length. Linformer approximates the attention matrix with a low-rank factorization, achieving O(n) complexity while maintaining model quality. This breakthrough enables Transformer models to scale to significantly longer sequences.

---

*Reading date: 2024*
*Note status: Completed*
