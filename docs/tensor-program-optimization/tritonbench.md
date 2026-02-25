# TritonBench: Benchmarking Large Language Model Capabilities for Generating Triton Operators

<div class="paper-meta" markdown>

**Authors**: Jianling Li, Shangzhan Li, Zhenye Gao, Qi Shi, Yuxuan Li, Zefan Wang, Jiacheng Huang, Haojie Wang, Jianrong Wang, Xu Han, Zhiyuan Liu, Maosong Sun  
**Institution**: Tianjin University, Harbin Institute of Technology, HKUST (Guangzhou), Tsinghua University  
**Conference**: arXiv 2025  
**Paper Link**: [arXiv:2502.14752](https://arxiv.org/abs/2502.14752)  
**GitHub**: [thunlp/TritonBench](https://github.com/thunlp/TritonBench)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Triton</span>
<span class="paper-tag">Benchmark</span>
<span class="paper-tag">LLM for Code</span>
<span class="paper-tag">Kernel Optimization</span>
<span class="paper-tag">GPU Programming</span>
</div>

## Abstract

TritonBench 是首个针对 Triton 算子生成的综合性 benchmark，用于系统评估 LLM 生成 Triton 代码的能力。与传统代码 benchmark 仅关注功能正确性不同，TritonBench 同时评估生成代码在 NVIDIA GPU 上的执行效率，更贴合工业需求。

**Key Contributions**:

- 提出双通道评估框架：TRITONBENCH-G（184 个来自 GitHub 的真实算子）和 TRITONBENCH-T（166 个对齐 PyTorch 接口的算子）
- 引入性能感知评估指标：除正确性外，还评估 Speed Up 和 GPU Efficiency
- 对主流 LLM 进行全面评估，揭示当前模型在 Triton 代码生成上的显著不足
- 开源 benchmark 框架及评估工具

## Core Ideas

### Benchmark 设计动机

Triton 作为高层 Python-like GPU 编程语言，在 vLLM、LightLLM、Liger-kernel、unsloth 等主流 LLM 框架中被广泛采用。然而：

- 编写高性能 Triton 算子仍需大量手动调优（指针算术、内存访问模式等）
- 现有 LLM 在通用代码生成上表现出色，但对 Triton 这类 DSL 缺乏规范理解和 GPU 编程经验
- 缺乏针对 Triton 的系统性评估框架——现有代码 benchmark 主要面向通用语言且仅关注正确性

### TRITONBENCH-G：真实世界算子通道

从 GitHub 爬取 Triton 相关仓库（>100 stars，共 95 个仓库、845 个 Python 文件），经过多轮筛选构建：

1. **数据收集**: Prompt-based 过滤 → 250 个候选文件 → 人工审查（补全缺失组件、去重、调试）→ 184 个高质量算子
2. **难度分级**: d1-d5 五个难度等级，由 LLM 初评 + 两位领域专家人工验证
3. **质量评估**: 平均 GPU Efficiency 43.0%，其中 19.6% 的专业开发者编写的算子 GPU 效率低于 10%
4. **测试设计**: 基于 tensor 的测试输入（非标量），平均每个算子 3.6 个测试分支

算子分布以高频操作为主：Attention (20.0%)、MatMul (10.9%)、LayerNorm (6.5%)、SoftMax (3.8%)

### TRITONBENCH-T：PyTorch 对齐算子通道

作为 TRITONBENCH-G 的补充，覆盖公开源中代表性不足的算子：

1. **构建流程**: 从 PyTorch (v2.6.0) 中筛选需要 GPU 交互的算子 → 按 The Stack V2 中的使用频率分为高频/低频各 40 个 → 组合融合为 166 个算子
2. **融合策略**: 常见算子组合、常见+非常见组合、非常见算子组合，确保前序算子输出可作为后续算子输入
3. **指令来源**: 直接从 PyTorch 文档提取（vs. TRITONBENCH-G 的 LLM 生成 + 专家验证）

### 评估指标体系

| 指标 | 说明 | 适用通道 |
|------|------|---------|
| Similarity (CodeBLEU) | 文本级相似度，N-gram/语法/数据流各权重 0.25 | G |
| Call Accuracy | 代码能否无错运行 | G, T |
| Execution Accuracy | 输入输出行为是否正确 | G, T |
| Speed Up | 正确执行算子相对参考实现的加速比 $\text{SpeedUp} = t_{\text{ref}} / t_{\text{gen}}$ | G, T |
| GPU Efficiency | GPU 资源利用率（实测 GB/s 或 TFLOPS / A100 理论峰值） | G |

## Experimental Results

### TRITONBENCH-G 主要结果

评估平台：NVIDIA A100。格式为 zero-shot / one-shot。

| Model | Size | Execution Accuracy | Speed Up | GPU Efficiency |
|-------|------|--------------------|----------|----------------|
| Qwen2.5-Coder | 7B | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| DeepSeek-Coder | 6.7B | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Qwen2.5-Coder-sft | 7B | 4.89 / 10.87 | 1.56 / 1.22 | 51.71 / 46.70 |
| DeepSeek-Coder-sft | 6.7B | 9.87 / 11.96 | 1.03 / 1.11 | 47.68 / 42.26 |
| GPT-4o | - | 10.33 / 16.84 | 0.97 / 1.19 | 48.80 / 53.33 |
| Claude-3.5-Sonnet | - | 9.79 / 19.57 | 0.90 / 1.54 | 59.31 / 49.32 |
| Qwen2.5-72B | 72B | 10.87 / 16.31 | 0.96 / 1.19 | 23.28 / 49.40 |
| DeepSeek-R1 | 685B | 13.05 / 22.83 | 1.11 / 1.22 | 44.83 / 46.70 |
| GPT-o1 | - | 14.23 / **23.91** | 0.92 / 1.14 | 54.25 / 46.37 |

### TRITONBENCH-T 主要结果

| Model | Size | Execution Accuracy | Speed Up |
|-------|------|--------------------|----------|
| Qwen2.5-Coder | 7B | 0.00 / 0.00 | 0.00 / 0.00 |
| DeepSeek-Coder | 6.7B | 0.00 / 1.81 | 0.00 / 0.94 |
| Qwen2.5-Coder-sft | 7B | 17.47 / 15.67 | 0.98 / 0.92 |
| DeepSeek-Coder-sft | 6.7B | 19.28 / 16.26 | 0.91 / 0.85 |
| GPT-4o | - | 36.75 / 32.53 | 0.98 / 0.94 |
| Claude-3.5-Sonnet | - | 29.52 / 33.70 | 0.93 / 0.89 |
| Qwen2.5-72B | 72B | 30.12 / 16.30 | 1.07 / 0.92 |
| DeepSeek-R1 | 685B | **53.01** / 45.78 | 1.03 / **1.91** |
| GPT-o1 | - | 32.53 / 43.37 | 1.21 / 1.10 |

### 关键发现

- **领域专用模型 vs 通用模型**: 未经微调的 7B 领域模型（Qwen2.5-Coder、DeepSeek-Coder）在 TRITONBENCH-G 上执行准确率为 0%；经 8K 语料微调后显著提升
- **最强模型**: DeepSeek-R1 和 GPT-o1 表现最优，GPT-o1 在 G 通道 one-shot 达到 23.91%，DeepSeek-R1 在 T 通道 zero-shot 达到 53.01%
- **One-shot 效果**: 高质量示例对 Triton 生成至关重要，G 通道 zero-shot → one-shot 提升约 10%
- **难度差异**: TRITONBENCH-T 整体比 G 简单，因 G 以 d3/d4 高难度算子为主
- **性能加速来源**: T 通道的加速主要来自算子融合，减少冗余内存读写

### 错误模式分析

论文将 16 种错误类型归为 4 大类：

| 错误类别 | 包含类型 |
|---------|---------|
| Syntax | SyntaxError, IndentationError |
| Attr&Type | AttributeError, TypeError, NotImplementedError |
| Name&Ref | NameError, KeyError, IndexError, ModuleNotFoundError, ImportError |
| Run&Logc | ValueError, RuntimeError, AssertionError, CompilationError 等 |

关键观察：One-shot 设置下 Syntax 和 Name&Ref 错误增加，但 Run&Logc 错误显著减少，说明示例有助于理解逻辑结构和 Triton 规范。DeepSeek-R1 对语法错误更鲁棒，GPT-o1 对逻辑错误处理更好。

## Limitations

- 所有评估仅在 NVIDIA A100 上进行，未覆盖其他 GPU 架构（如 H100、AMD MI300X）
- Benchmark 规模有限（G: 184, T: 166），可能无法完全代表 Triton 算子开发的全部场景
- TRITONBENCH-G 以高频算子为主（Attention 20%、MatMul 10.9%），长尾算子覆盖不足
- 评估仅考虑单次生成，未探索迭代优化或 agentic 工作流的效果
- 微调数据仅 8K 样本，更大规模的领域数据可能带来更显著的提升

---

*Reading date: 2025-02*
*Note status: Completed*
