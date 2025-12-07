# DLRM: Deep Learning Recommendation Model for Personalization and Recommendation Systems

<div class="paper-meta" markdown>

**Authors**: Maxim Naumov et al.
**Institution**: Facebook AI Research
**Conference**: arXiv 2019
**Paper Link**: [arXiv:1906.00091](https://arxiv.org/abs/1906.00091)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Recommendation Systems</span>
<span class="paper-tag">Deep Learning</span>
<span class="paper-tag">Embedding Tables</span>
</div>

## Abstract

DLRM is a state-of-the-art deep learning recommendation model developed by Facebook that handles both dense and sparse features. It uses specialized parallelization schemes with model parallelism on embedding tables and data parallelism on fully-connected layers.

**Key Contributions**:
- Novel interaction layer mimicking factorization machines
- Specialized parallelization scheme for large-scale recommendation
- Open-source implementation in PyTorch and Caffe2

## Core Ideas

DLRM specifically interacts embeddings in a structured way that significantly reduces model dimensionality by only considering cross-terms produced by dot-products between pairs of embeddings. The model has become part of the popular MLPerf Benchmark and is widely used in production recommendation systems.

---

*Reading date: 2024*
*Note status: Completed*
