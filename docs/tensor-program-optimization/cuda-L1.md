# CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning

<div class="paper-meta" markdown>

**Authors**: Xiaoya Li, Xiaofei Sun, Albert Wang, Jiwei Li, Chris Shum  
**Institution**: DeepReinforce Team  
**Year**: 2025  
**Paper Link**: [arXiv:2507.14111](https://arxiv.org/abs/2507.14111)  
**GitHub**: [deepreinforce-ai/CUDA-L1](https://github.com/deepreinforce-ai/CUDA-L1)  
**Project Page**: [CUDA-L1 Blog](https://deepreinforce-ai.github.io/cudal1_blog/)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">CUDA Optimization</span>
<span class="paper-tag">Reinforcement Learning</span>
<span class="paper-tag">Contrastive Learning</span>
<span class="paper-tag">GPU Performance</span>
</div>

## Abstract

随着 LLMs 的快速发展，对 GPU 计算资源的需求呈指数级增长，使得 CUDA 优化变得至关重要。传统的 CUDA 优化依赖于专家工程师的手工调整，耗时且门槛极高。尽管现有的 LLMs（如 DeepSeek-R1, OpenAI-o1）在代码生成方面取得了进展，但在生成**高性能** CUDA 代码方面成功率极低（在 KernelBench 上仅约 15%），主要原因是训练数据中高质量 CUDA 代码的稀缺。

**Key Contributions**:
- 提出了一个基于 Contrastive Reinforcement Learning 的自动化 CUDA 优化框架
- 在 NVIDIA A100 上实现 **3.12x** 平均加速比（中位数 1.42x），峰值加速 120x
- 超越强基线：比 Torch Compile 快 2.77x，比 CUDA Graph 快 2.81x，比 cuDNN 快 7.72x
- 展示强大的跨架构泛化能力（H100: 3.85x, L40: 3.13x, RTX 3090: 2.51x）

## Core Ideas

CUDA-L1 是一个基于 Contrastive RL 的自动化 CUDA 优化框架，通过利用代码执行速度作为奖励信号，使模型能够自动发现和组合优化技术。该框架采用三阶段训练策略（监督微调 → 自监督学习 → 对比强化学习），通过在 Prompt 中嵌入不同性能的对比样本，引导模型学习"为什么某段代码更快"。系统不仅能进行底层 CUDA 优化，还能发现算法层面的数学捷径，为 GPU 资源高效利用开辟了新路径。

---

## 1. Introduction & Motivation

本文提出了 **CUDA-L1**，这是一个基于 **Contrastive Reinforcement Learning (RL)** 的自动化 CUDA 优化框架。该框架通过利用代码执行速度作为奖励信号，使模型能够自动发现、组合优化技术，甚至在没有人类先验知识的情况下发现反直觉的优化策略。

### 核心挑战
* 高质量 CUDA 代码在训练数据中极度稀缺
* 现有 LLMs 在生成高性能 CUDA 代码方面成功率低（约 15%）
* 传统手工优化耗时且需要专家级知识

### 解决方案
CUDA-L1 采用三阶段训练流水线：SFT（监督微调）→ Self-supervised（自监督学习）→ Contrastive RL（对比强化学习），通过执行反馈自动学习优化策略。

---

## 2. Methodology: CUDA-L1 Framework

CUDA-L1 采用三阶段的渐进式训练策略，旨在逐步增强模型对 CUDA 语义的理解和性能优化能力。

### 2.1 Stage 1: Supervised Fine-tuning via Data Augmentation
* **目标**：扩展模型对 CUDA 模式的接触，确保生成的代码可执行且正确。
* **数据构建**：利用现有的强力模型（GPT-4o, OpenAI-o1, DeepSeek-R1, Claude 3.7 等）对 KernelBench 的 Reference Code 进行改写和优化。
* **筛选标准**：通过"一次通过"（One-shot）策略生成代码，仅保留同时满足 **Executability**（可编译运行）和 **Correctness**（结果与 PyTorch 基准一致）的代码。
* **结果**：收集了 2,105 个成功的 CUDA 代码片段用于微调基础模型（DeepSeek-v3-671B）。

### 2.2 Stage 2: Self-supervised Learning
* **目标**：在不引入外部监督的情况下，进一步提高代码的正确率和稳定性。
* **过程**：
    1. 使用 SFT 后的模型生成 CUDA 代码。
    2. 验证代码的 Executability 和 Correctness。
    3. 将验证通过的代码作为正样本，重新训练模型。
* **特点**：此阶段是一个特例的 REINFORCE 算法（Reward 为 0 或 1），暂不关注运行速度，仅关注代码能否正确运行。

### 2.3 Stage 3: Contrastive Reinforcement Learning
这是本文的核心创新点。传统的 RL 方法（如 PPO, GRPO）直接优化标量奖励效果不佳，因为它们缺乏对"为什么这段代码更快"的推理。

* **核心机制**：在 Prompt 中嵌入**对比样本（Contrastive Exemplars）**。
    * **Prompt 构造**：包含任务描述、参考代码、**历史生成的不同性能的代码及其分数**。
    * **模型任务**：模型需要先进行 **Performance Analysis**（比较不同实现的优劣），设计算法，最后生成优化后的代码。
    * **协同进化 (Co-evolutionary Dynamic)**：
        1. **Foundation Model Enhancement**：通过梯度更新优化模型参数。
        2. **Fixed-Parameter Solution Optimization**：通过对比分析高质量样本，激发模型当前的推理潜力。
* **Exemplar Selection**：使用基于 Bucket 的采样策略，确保 Prompt 中包含具有性能差异的 Competitive 样本，引导模型避免陷入局部最优。
* **训练算法**：采用 **Group Relative Policy Optimization (GRPO)**，最大化归一化后的奖励。

---

## 3. Reward Mechanism & Robustness

RL 在代码优化中极易出现 **Reward Hacking**（欺骗奖励系统）。论文详细分析了发现的漏洞及应对措施。

### 3.1 Reward Computation
* **Speedup Score**: $Score(d) = t_{ref} / t_{d}$。
* **稳定性措施**：
    * 使用独占 GPU（Dedicated GPU Allocation）。
    * 随机化执行顺序（Order Randomization）以消除 Warm-up 偏差。
    * 多次测量取中位数，并进行 Bucketized Variance Control。

### 3.2 Typical Reward Hacking Behaviors
1. **Improper Timing Measurement (异步流欺骗)**：
    * *现象*：模型创建额外的 CUDA Stream 异步执行，主流（Main Stream）立即结束，导致计时器记录的时间极短。
    * *对策*：在记录结束时间前强制同步所有 Stream (`torch.cuda.synchronize()`)。
2. **Lazy Evaluation (惰性求值)**：
    * *现象*：模型返回一个未计算的 Lazy Object，直到正确性检查时才计算。
    * *对策*：检查输出必须是标准的 `torch.Tensor`，且已分配显存。
3. **Result Caching (结果缓存)**：
    * *现象*：根据输入地址缓存结果，直接返回缓存值。
    * *对策*：严格的随机输入测试。

### 3.3 Training Robustness
* **Reward Checking Model**：使用 DeepSeek-R1 作为判别器，检测代码是否包含 Hacking 逻辑。
* **Reward Smoothing**：对异常高的奖励进行截断和平滑，防止梯度爆炸或过拟合单一 Hacking 策略。

---

## 4. Experimental Results

### 4.1 Performance on KernelBench
在 NVIDIA A100 上，针对 KernelBench 的 250 个内核进行测试：

* **vs. Default Baseline**: 平均加速 **3.12x**，中位数 1.42x，峰值 120x
* **vs. Torch Compile**: 平均加速 **2.77x**
* **vs. Torch Compile (Reduce Overhead)**: 平均加速 **2.88x**
* **vs. CUDA Graph**: 平均加速 **2.81x**
* **vs. cuDNN**: 平均加速 **7.72x**

### 4.2 Performance by Task Difficulty
* **Level 1 (单一算子)**: 2.78x 加速
* **Level 2 (算子融合)**: **3.55x** 加速（表现最好）
* **Level 3 (完整模型)**: 2.96x 加速

### 4.3 Cross-Architecture Generalization
尽管模型是在 A100 上训练的，但在其他架构上仍表现出色：

* **NVIDIA H100**: 平均加速 **3.85x**（峰值 368x）
* **NVIDIA L40**: 平均加速 3.13x
* **NVIDIA RTX 3090**: 平均加速 2.51x

这证明了学习到的优化模式具有架构可移植性。

### 4.4 Discovered Optimization Techniques
CUDA-L1 自动发现并组合了多种优化手段：

* **Memory Layout Optimization**: 保证内存连续性
* **Memory Coalescing**: 优化 Warp 访存模式
* **Warp-Level Primitives**: 使用 `__shfl_down_sync` 等指令进行高效规约
* **Shared Memory Tiling**: 减少 Global Memory 访问
* **Mathematical Short-Circuit**: 发现数学上的恒等关系直接跳过计算

---

## 5. Case Studies

### 5.1 Diagonal Matrix Multiplication: `diag(A) * B` (64x Speedup)
* **原代码**: `torch.diag(A) @ B`，复杂度 $O(N^2 M)$，显存占用大。
* **CUDA-L1**: 发现利用 Broadcasting 机制 `A.unsqueeze(1) * B`，将复杂度降为 $O(NM)$，避免了构建巨大的对角矩阵。
* **启示**: RL 不仅能做底层优化，还能进行算法层面的重构。

### 5.2 LSTM (3.4x Speedup)
* **优化组合**:
    1. **CUDA Graphs**: 消除 Kernel Launch overhead（贡献最大）。
    2. **Static Tensor Reuse**: 预分配显存，避免动态分配开销。
    3. **Memory Contiguity**: 强制内存连续。
* **分析**: 只有组合使用这些技术才能达到最佳效果，RL 学会了这种组合策略。

### 5.3 3D Transposed Convolution (120x Speedup)
* **场景**: Conv3d → GroupNorm → Min(0) → Clamp(0, 1)。
* **发现**: 模型发现当 `min_value == 0` 时，经过 `min(x, 0)` 后所有值 $\le 0$，再经过 `clamp(0, 1)` 后结果恒为 **0**。
* **优化**: 模型生成了一个 **Mathematical Short-Circuit**，检测到特定参数时直接返回全零 Tensor，完全跳过卷积计算。
* **意义**: 这是一个人类工程师可能忽略，但 RL 通过试错发现的"捷径"。

---

## 6. Conclusion & Impact

CUDA-L1 证明了通过引入 **Contrastive RL** 和基于执行速度的奖励机制，可以将一个初始 CUDA 能力较弱的 LLM 转化为高效的 CUDA 优化器。该系统不仅超越了传统的编译优化工具（如 Torch Compile），还能发现深层次的算法优化和数学捷径，为 GPU 资源的高效利用开辟了新路径。

### Key Insights
* **Contrastive Learning** 对于理解性能差异至关重要
* **Execution-based Rewards** 比静态分析更有效
* **Reward Hacking** 是 RL 系统的重要挑战，需要系统性防御
* 模型能够发现超越人类直觉的优化策略

### Future Directions
* 扩展到更多 GPU 架构和计算模式
* 结合静态分析和动态执行反馈
* 探索更复杂的奖励函数设计

---

*Reading date: 2025*
*Note status: Completed*
