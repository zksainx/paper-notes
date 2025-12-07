# NeuSight: Forecasting GPU Performance for Deep Learning Training and Inference

<div class="paper-meta" markdown>

**Authors**: Research Team
**Institution**: Multiple Institutions
**Conference**: ASPLOS 2025
**Paper Link**: [arXiv:2407.13853](https://arxiv.org/abs/2407.13853)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">GPU Performance</span>
<span class="paper-tag">Deep Learning</span>
<span class="paper-tag">Performance Forecasting</span>
</div>

## Abstract

NeuSight is a forecasting framework that predicts the performance of diverse deep learning models for both training and inference on unseen GPUs without requiring actual execution on the target GPU. It achieves exceptional accuracy compared to previous approaches.

**Key Contributions**:
- Reduces latency prediction error from 121.4% and 30.8% to 2.3% for GPT-3 on H100
- Inference error: 9.7% (vs prior work 220.9%), Training error: 7.3% (vs prior work 725.8%)
- Tile-based approach predicting device utilization per tile
- Works for both single-GPU and multi-GPU distributed settings

## Core Ideas

Unlike prior work that directly predicts kernel latency using machine learning, NeuSight distills kernel execution into tiles, predicts utilization per tile, and determines latency based on GPU architecture performance bounds. This novel approach significantly improves accuracy across various workloads and modern GPUs.

---

*Reading date: 2024*
*Note status: Completed*
