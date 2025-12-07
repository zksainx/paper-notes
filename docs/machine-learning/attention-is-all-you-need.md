# Attention Is All You Need

<div class="paper-meta" markdown>

**作者**: Ashish Vaswani, Noam Shazeer, Niki Parmar, et al.
**机构**: Google Brain, Google Research
**会议**: NeurIPS 2017
**论文链接**: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Transformer</span>
<span class="paper-tag">注意力机制</span>
<span class="paper-tag">序列建模</span>
</div>

## 📝 摘要

本文提出了 Transformer 架构，这是一个完全基于注意力机制的序列转换模型，摒弃了循环和卷积结构。在机器翻译任务上，Transformer 不仅取得了更好的性能，而且训练速度显著提升。

**主要贡献**：
- 提出了纯注意力架构，无需循环或卷积
- 引入多头自注意力（Multi-Head Self-Attention）机制
- 在 WMT 2014 英德和英法翻译任务上达到 SOTA

## 💡 核心思想

### 1. 自注意力机制（Self-Attention）

通过计算序列中每个位置与其他所有位置的关联度，捕获长距离依赖关系：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中 $Q$（Query）、$K$（Key）、$V$（Value）是输入的线性变换。

### 2. 多头注意力（Multi-Head Attention）

并行使用多个注意力头，每个头学习不同的表示子空间：

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O
$$

### 3. 位置编码（Positional Encoding）

由于模型没有循环结构，需要额外添加位置信息：

$$
\begin{align}
PE_{(pos, 2i)} &= \sin(pos / 10000^{2i/d_{model}}) \\
PE_{(pos, 2i+1)} &= \cos(pos / 10000^{2i/d_{model}})
\end{align}
$$

## 🏗️ 模型架构

Transformer 采用编码器-解码器结构：

**编码器**：
- 6 层堆叠
- 每层包含：多头自注意力 + 前馈网络
- 残差连接 + 层归一化

**解码器**：
- 6 层堆叠
- 每层包含：掩码多头自注意力 + 编码器-解码器注意力 + 前馈网络
- 残差连接 + 层归一化

## 📊 实验结果

### 机器翻译性能

| 模型 | WMT'14 EN-DE | WMT'14 EN-FR |
|------|--------------|--------------|
| **Transformer (big)** | **28.4 BLEU** | **41.8 BLEU** |
| ConvS2S | 25.2 BLEU | 40.5 BLEU |

### 训练效率

- **训练时间**：在 8 个 P100 GPU 上训练 3.5 天（base 模型）
- **并行化**：相比 RNN 可以更好地并行化，训练速度快数倍

## 🤔 个人笔记

### 优势

1. **并行化能力强**：不像 RNN 需要顺序处理，可以并行计算所有位置
2. **长距离依赖**：通过自注意力直接建模任意距离的依赖关系
3. **可解释性**：注意力权重可以可视化，了解模型关注哪些部分

### 局限性

1. **计算复杂度**：自注意力的复杂度是 $O(n^2)$，序列越长计算量越大
2. **位置编码**：固定的正弦位置编码可能不是最优选择
3. **小数据集**：在数据量较小时，Transformer 可能不如 RNN

### 影响与启发

- **NLP 革命**：Transformer 成为 BERT、GPT 等预训练模型的基础
- **跨领域应用**：Vision Transformer (ViT) 将其成功应用于计算机视觉
- **研究方向**：如何降低自注意力的复杂度成为重要研究方向（Linformer、Performer 等）

## 🔗 相关论文

- [BERT: Pre-training of Deep Bidirectional Transformers](../nlp/bert.md) - 基于 Transformer 的预训练模型
- **GPT Series** - 生成式预训练 Transformer
- **Vision Transformer (ViT)** - Transformer 在视觉领域的应用

## 📌 关键代码片段

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections and reshape
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)

        # Concatenate heads and apply final linear
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)
```

---

*阅读日期：2024年*
*笔记状态：已完成*
