# KernelBlaster: Continual Cross-Task CUDA Optimization via Memory-Augmented In-Context Reinforcement Learning

<div class="paper-meta" markdown>

**Authors**: Kris Shengjun Dong, Sahil Modi, Dima Nikiforov, Sana Damani, Edward Lin, Siva Kumar Sastry Hari, Christos Kozyrakis  
**Institution**: NVIDIA, University of California Berkeley  
**Conference**: arXiv 2026  
**Paper Link**: [arXiv:2602.14293](https://arxiv.org/abs/2602.14293)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">CUDA</span>
<span class="paper-tag">Reinforcement Learning</span>
<span class="paper-tag">LLM for Code</span>
<span class="paper-tag">Kernel Optimization</span>
<span class="paper-tag">Cross-Task Learning</span>
</div>

## Abstract

KernelBlaster 提出了 MAIC-RL（Memory-Augmented In-context Reinforcement Learning）框架，用于自动化 CUDA 代码优化。与以往独立优化每个 kernel 的方法不同，KernelBlaster 通过构建可检索的 Persistent CUDA Knowledge Base，使 LLM agent 能够从历史优化经验中学习，并在未来任务中做出系统性的决策。框架采用 profile-guided 的 textual gradient 机制，在不更新模型权重的情况下实现推理时学习。

**Key Contributions**:

- 提出 In-Context Reinforcement Learning (ICRL) 框架，将 CUDA 优化问题建模为 RL 问题，使用 textual gradient 从 profiling 数据中捕获语义信息，实现推理时学习
- 构建 Persistent CUDA Knowledge Base，采用层次化表示将代码实例按性能状态分类，实现高效的优化候选遍历
- 实现跨任务长期学习，在生成优化 kernel 的同时聚合跨问题知识，生成可复用的 artifact，加速新问题和新 GPU 平台上的收敛
- 开源 textual RL agentic 框架，包含基线 CUDA kernel、初始化数据库和测试工具

## Core Ideas

### MAIC-RL 框架概览

KernelBlaster 的核心是一个 agentic 工作流，由三个关键组件构成：

1. **LLM Agent**: 负责生成和优化 CUDA 代码，基于 profiling 反馈和知识库检索结果进行迭代改进
2. **Profile-Guided Feedback**: 从 GPU profiler 收集硬件级性能指标，转化为 textual gradient 指导优化方向
3. **Persistent CUDA Knowledge Base**: 存储和组织历史优化经验，支持跨任务知识检索和复用

整体流程：Agent 接收优化任务 → 从知识库检索相关经验 → 生成候选优化代码 → Profiling 评估 → Textual gradient 反馈 → 迭代优化 → 将新知识写回知识库。

### In-Context Reinforcement Learning (ICRL)

传统 RL 方法通过反向传播更新模型权重来学习策略，成本高昂且难以泛化。KernelBlaster 采用 ICRL 方法：

- **Textual Gradient**: 将 profiling 数据（如内存带宽利用率、计算吞吐量、occupancy 等）转化为自然语言描述的优化建议，作为"梯度"信号指导 LLM 的下一步优化
- **推理时学习**: 所有学习发生在 LLM 的上下文窗口内，无需微调或权重更新，相比传统 RL 方法收敛更快
- **语义丰富性**: Textual gradient 能够捕获比数值 reward 更丰富的语义信息，例如"shared memory bank conflict 导致带宽下降"比单纯的 speedup 数值提供更多可操作的优化方向

### Persistent CUDA Knowledge Base

知识库采用层次化结构组织优化经验：

- **Performance State 分类**: 将代码实例按性能表现分为不同状态（如 baseline、partially optimized、highly optimized），形成可扩展的表示
- **层次化表示**: 在有限的模型上下文中高效利用空间，避免将所有历史代码直接放入 prompt
- **检索机制**: 根据当前任务特征（算子类型、数据规模、硬件目标等）检索最相关的历史优化经验
- **持久化**: 知识库作为独立 artifact 存在，可跨会话、跨任务持续积累

### Profile-Guided Textual Gradient Flow

优化流程的核心循环：

```
1. 生成/修改 CUDA kernel
2. 编译并在目标 GPU 上执行
3. 收集 profiling 数据（内存吞吐、计算利用率、occupancy 等）
4. 将 profiling 数据转化为 textual gradient（自然语言优化建议）
5. LLM agent 基于 textual gradient 生成下一版本优化代码
6. 重复直到收敛或达到迭代上限
```

这种 profile-guided 方法使 agent 能够系统性地探索高潜力优化策略，而非简单的代码重写。

### 跨任务学习与硬件适配

KernelBlaster 的独特之处在于同时完成两个目标：

- **即时优化**: 为当前任务生成高性能 CUDA kernel
- **知识积累**: 将优化过程中获得的经验聚合到知识库中

知识库可针对不同应用领域和 GPU 架构进行特化（如 NVIDIA Ampere vs. Hopper），使 agent 能够将积累的经验有效应用于未来未见过的问题。这解决了手动优化的可扩展性瓶颈——例如 FlashAttention-2 移植到 H100 时性能下降约 47%，需要重新设计才能恢复效率（FlashAttention-3）。

## Experimental Results

**Benchmark**: KernelBench (Level 1/2/3)

**主要结果**（相对 PyTorch baseline 的几何平均加速比）：

| Benchmark Level | 任务类型 | Geometric Mean Speedup |
|----------------|---------|----------------------|
| KernelBench L1 | 单算子优化 | 1.43x |
| KernelBench L2 | 复杂算子组合 | 2.50x |
| KernelBench L3 | 完整模型加速 | 1.50x |

**关键观察**:

- L2 上取得最大加速比（2.50x），表明框架在处理复杂算子组合时优势明显
- 跨任务知识积累使 agent 能够系统性地探索高潜力优化策略，超越简单的代码重写
- Profile-guided textual gradient 机制引导 agent 关注真正的性能瓶颈

## Limitations

- 框架依赖高质量的 profiling 反馈，profiler 的精度和覆盖范围直接影响优化效果
- 知识库的检索质量取决于任务特征的表示和匹配算法
- 论文主要在 KernelBench 上评估，对更广泛的实际工作负载的泛化能力有待验证
- 开源代码库尚未在论文发布时同步释出（将在后续版本中发布）

---

*Reading date: 2026-02*
*Note status: Completed*
