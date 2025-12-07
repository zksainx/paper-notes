# MetaTune: Meta-Learning Based Cost Model for Fast and Efficient Auto-tuning Frameworks

<div class="paper-meta" markdown>

**Authors**: Jaehun Ryu et al.
**Institution**: Multiple Institutions
**Conference**: arXiv 2021
**Paper Link**: [arXiv:2102.04199](https://arxiv.org/abs/2102.04199)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Meta-Learning</span>
<span class="paper-tag">Cost Model</span>
<span class="paper-tag">TVM</span>
</div>

## Abstract

MetaTune is a meta-learning based cost model that quickly and accurately predicts the performance of optimized codes with pre-trained model parameters. It addresses the large space exploration and cost model training overheads in auto-tuning frameworks.

**Key Contributions**:
- GNN-based meta-learning for cost model training
- Encodes convolution kernels as structurally similar graphs
- 8-13% better inference time on average for four CNN models
- Outperforms transfer learning by 10% in cross-platform cases

## Core Ideas

MetaTune encodes convolution kernel codes as graphs to facilitate meta-learning, meta-trains a GNN model with minimal input data, and predicts optimization parameters for unseen operations during compilation. Implemented as an alternative cost model for TVM's auto-tuning framework, it provides comparable or lower optimization time while improving inference performance.

---

*Reading date: 2024*
*Note status: Completed*
