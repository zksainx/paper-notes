# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

<div class="paper-meta" markdown>

**作者**: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
**机构**: Google AI Language
**会议**: NAACL 2019
**论文链接**: [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">BERT</span>
<span class="paper-tag">预训练</span>
<span class="paper-tag">双向Transformer</span>
<span class="paper-tag">语言模型</span>
</div>

## 📝 摘要

BERT（Bidirectional Encoder Representations from Transformers）是一个基于 Transformer 的双向预训练语言模型。通过在大规模无标注文本上进行预训练，然后在下游任务上微调，BERT 在 11 个 NLP 任务上刷新了 SOTA 记录。

**主要贡献**：
- 提出掩码语言模型（MLM）实现真正的双向预训练
- 引入下一句预测（NSP）任务学习句子关系
- 在多个 NLP 任务上取得显著提升

## 💡 核心思想

### 1. 双向上下文建模

与传统的单向语言模型（如 GPT）不同，BERT 通过掩码语言模型同时利用左右上下文：

```
传统 LM:  The cat [sat] on the mat
           ←←←←←
GPT:      The cat [sat] on the mat
           →→→→→
BERT:     The cat [MASK] on the mat
           ←←←→→→→
```

### 2. 掩码语言模型（Masked Language Model）

随机掩盖输入中 15% 的词，训练模型预测被掩盖的词：

- 80% 替换为 `[MASK]`
- 10% 替换为随机词
- 10% 保持不变

**目标**：最大化被掩盖词的条件概率

### 3. 下一句预测（Next Sentence Prediction）

给定句子对 (A, B)，预测 B 是否是 A 的下一句：

```
Input:  [CLS] Sentence A [SEP] Sentence B [SEP]
Label:  IsNext / NotNext
```

## 🏗️ 模型架构

BERT 基于 Transformer 编码器，有两个版本：

| 模型 | 层数 | 隐藏维度 | 注意力头 | 参数量 |
|------|------|----------|----------|--------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

### 输入表示

三种嵌入求和：

```
Token Embeddings:    [CLS] my dog is cute [SEP] he likes play ##ing [SEP]
Segment Embeddings:  E_A   E_A E_A E_A E_A E_A  E_B  E_B   E_B  E_B   E_B
Position Embeddings: E_0   E_1 E_2 E_3 E_4 E_5  E_6  E_7   E_8  E_9   E_10
```

### 特殊标记

- `[CLS]`：序列分类标记，其输出用于分类任务
- `[SEP]`：句子分隔符
- `[MASK]`：掩码标记

## 📊 实验结果

### GLUE 基准测试

| 任务 | 指标 | BERT-Base | BERT-Large | 之前 SOTA |
|------|------|-----------|------------|-----------|
| MNLI | Acc | 84.6/83.4 | **86.7/85.9** | 80.5/80.1 |
| QQP | F1 | 71.2 | **72.1** | 66.1 |
| QNLI | Acc | 90.5 | **92.7** | 87.4 |
| SST-2 | Acc | 93.5 | **94.9** | 93.2 |
| CoLA | Matthews Corr | 52.1 | **60.5** | 35.0 |

### SQuAD 问答

| 模型 | SQuAD 1.1 F1 | SQuAD 2.0 F1 |
|------|--------------|--------------|
| BERT-Base | 88.5 | 76.3 |
| **BERT-Large** | **93.2** | **83.1** |
| 人类表现 | 91.2 | 86.8 |

BERT-Large 在 SQuAD 1.1 上超越人类表现！

### 命名实体识别（CoNLL-2003 NER）

| 模型 | F1 |
|------|----|
| 之前 SOTA | 92.6 |
| **BERT-Large** | **92.8** |

## 🤔 个人笔记

### 关键设计

1. **WordPiece 分词**：处理未登录词，减小词表大小
2. **Segment Embeddings**：区分句子对中的不同句子
3. **预训练 + 微调范式**：通用预训练 → 任务特定微调

### 为什么有效？

- **大规模预训练**：从 BooksCorpus (800M 词) + Wikipedia (2,500M 词) 学习通用语言表示
- **双向建模**：比单向模型能捕获更丰富的上下文
- **深度 Transformer**：强大的表示学习能力

### 局限性

1. **计算成本高**：预训练需要大量计算资源（4-16 TPU，几天时间）
2. **NSP 任务争议**：后续研究（RoBERTa）表明 NSP 可能不必要
3. **[MASK] 不一致**：预训练用 [MASK]，但微调时没有
4. **最大长度限制**：512 tokens 限制了长文本处理

### 后续发展

- **RoBERTa**：移除 NSP，动态掩码，更大批次
- **ALBERT**：参数共享，句子顺序预测（SOP）
- **ELECTRA**：判别式预训练，更高效
- **T5**：统一 text-to-text 框架
- **GPT-3/ChatGPT**：大规模生成式预训练

### 应用技巧

**下游任务微调**：
- 分类任务：使用 [CLS] 输出 + 线性层
- 序列标注：每个 token 输出 + 线性层
- 问答任务：预测答案起始和结束位置

**超参数**：
- Batch size: 16, 32
- Learning rate: 5e-5, 3e-5, 2e-5
- Epochs: 2-4

## 🔗 相关论文

- [Attention Is All You Need](../machine-learning/attention-is-all-you-need.md) - Transformer 架构基础
- **ELMo**: Deep contextualized word representations - 早期的双向预训练
- **GPT**: Improving Language Understanding by Generative Pre-Training - 单向预训练
- **RoBERTa**: A Robustly Optimized BERT Pretraining Approach - BERT 改进
- **ELECTRA**: Pre-training Text Encoders as Discriminators - 更高效的预训练

## 📌 关键代码片段

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载预训练模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 准备输入
text = "BERT is a powerful pre-trained model."
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# 前向传播
outputs = model(**inputs)
logits = outputs.logits
predictions = torch.argmax(logits, dim=-1)

print(f"Predictions: {predictions}")
```

### 微调示例

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# 加载数据集
dataset = load_dataset('glue', 'sst2')

# 准备模型
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 训练配置
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
)

trainer.train()
```

---

*阅读日期：2024年*
*笔记状态：已完成*
