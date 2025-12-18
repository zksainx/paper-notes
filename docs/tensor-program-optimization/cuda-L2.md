# CUDA-L2: Surpassing cuBLAS Performance for Matrix Multiplication through Reinforcement Learning

<div class="paper-meta" markdown>

**Authors**: Songqiao Su, Xiaofei Sun, Xiaoya Li, Albert Wang, Jiwei Li, Chris Shum  
**Institution**: DeepReinforce Team  
**Year**: 2025  
**Paper Link**: [arXiv:2512.02551](https://arxiv.org/abs/2512.02551)  
**GitHub**: [deepreinforce-ai/CUDA-L2](https://github.com/deepreinforce-ai/CUDA-L2)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">CUDA Optimization</span>
<span class="paper-tag">Matrix Multiplication</span>
<span class="paper-tag">Reinforcement Learning</span>
<span class="paper-tag">HGEMM</span>
<span class="paper-tag">cuBLAS / cuBLASLt</span>
</div>

---

## Abstract

Matrix Multiplication (MatMul) 是 LLM 训练与推理中最核心、最耗时的算子之一，其 CUDA Kernel 也是 GPU 上**人工优化程度最高**的计算单元。尽管 NVIDIA 提供了高度优化的闭源库（cuBLAS、cuBLASLt），在大规模、多维度配置空间（M, N, K）下，**人工调参难以系统性覆盖所有最优解**。

本文提出 **CUDA-L2**，一个基于 **LLM + Reinforcement Learning (RL)** 的自动化 HGEMM（Half-precision GEMM）优化系统。CUDA-L2 在 **1,000 种 (M, N, K) 配置**上系统性探索 kernel 设计空间，在 **NVIDIA A100** 上 **显著超越 cuBLAS 与 cuBLASLt（包括 AutoTuning）**，首次证明：  
> 即便是高度专家化、接近硬件极限的 HGEMM Kernel，仍可通过 LLM-guided RL 获得稳定、可复现的性能提升。

---

## Core Ideas

CUDA-L2 是 CUDA-L1 在 **高难度、强结构算子（HGEMM）** 场景下的系统性扩展，其核心思想包括：

1. **将 CUDA kernel 优化视为连续决策问题**，使用 execution speed 作为 RL reward；
2. **从通用 kernel → 专用 HGEMM kernel 的分阶段 RL 训练**；
3. **引入 Nsight Compute (NCU) profiling 指标作为上下文信息**，避免纯黑盒优化；
4. **通过 Contrastive RL 学习“为什么这个 kernel 更快”**，而非仅拟合 reward。

---

## 1. Introduction & Motivation

### 1.1 Problem Background

* HGEMM 在 Attention 与 FFN 中占据主要计算比例；
* 不同 (M, N, K)、layout（NN / TN）、accumulator precision 会导致**完全不同的最优 kernel**；
* cuBLAS / cuBLASLt 已高度优化，但：
  * 依赖 heuristic 或有限 AutoTuning；
  * 难以跨配置、跨场景系统性最优。

现实证据：即便专家团队（如 TensorRT-LLM）针对单一模型精调，仍可获得 **~13% 提升**。

### 1.2 Research Question

> 是否可以通过自动化方法，在**无需人工枚举与专家经验**的前提下，系统性超越 cuBLAS / cuBLASLt？

CUDA-L2 给出的答案是 **Yes**。

---

## 2. Preliminaries

### 2.1 HGEMM Definition

* 输入：  
  * A ∈ ℝ<sup>M×K</sup>, B ∈ ℝ<sup>K×N</sup>  
* 输出：  
  * C = AB（FP16 input，Tensor Core acceleration）
* Kernel 采用典型 **tiled GEMM** 结构：
  1. Global → Shared Memory
  2. Shared → Registers
  3. Tensor Core MMA
  4. Epilogue 写回

### 2.2 Baselines

CUDA-L2 对比以下主流实现：

* **torch.matmul**
* **cuBLAS**
  * NN / TN / max(NN, TN)
* **cuBLASLt**
  * heuristic
  * AutoTuning（≤100 candidates）

---

## 3. CUDA-L2 Framework

### 3.1 Relation to CUDA-L1

CUDA-L2 继承 CUDA-L1 的三段式结构，但针对 HGEMM 做出关键增强：

| 维度 | CUDA-L1 | CUDA-L2 |
|---|---|---|
| Kernel 类型 | KernelBench | 专注 HGEMM |
| 训练数据 | Benchmark kernel | 大规模 CUDA library |
| Context | Execution time | + NCU profiling |
| 搜索空间 | 小 | 极大（MNK × tile × pipeline） |

---

### 3.2 Training Pipeline

#### 3.2.1 Continued Pretraining

* 数据来源：
  * PyTorch / ATen / CUTLASS
  * NVIDIA 官方示例
  * Web CUDA code
* 构造：
  * Instruction + Retrieved Context + CUDA code
* 目标：
  * 补齐模型对 **现代 CUDA / CuTe / CUTLASS** 的知识缺口

---

#### 3.2.2 General Kernel RL

* 覆盖线性代数、Conv、Reduction、Attention 等；
* Reward：平均 speedup；
* 算法：**Contrastive RL + GRPO**；
* 目的：建立通用 CUDA 优化能力。

---

#### 3.2.3 HGEMM-Specific RL

这是 CUDA-L2 的**核心阶段**。

**Reward 定义**：

\[
r = \frac{1}{N} \sum_{i=1}^N \left(\frac{t_{ref}}{t_{custom}} - \alpha \cdot diff_i \right) - \beta \cdot L
\]

其中：
* `diff`：与 FP32 reference 的最大误差；
* `L`：代码长度惩罚；
* Context 包含 **NCU metrics**（SM occupancy、memory throughput、cache hit）。

---

## 4. Experimental Results

### 4.1 Overall Performance (A100)

**Offline Mode**（back-to-back execution）：

* +22.0% vs torch.matmul
* +19.2% vs cuBLAS-max
* +16.8% vs cuBLASLt-heuristic
* **+11.4% vs cuBLASLt-AutoTuning**

**Server Mode**（real-time inference）：

* 提升进一步扩大至 **+15.9% ~ +28.7%**

---

### 4.2 Effect of Problem Size

* 小矩阵（GPU 未饱和）：**1.3–1.4×**
* 大矩阵（compute-bound）：趋近 1.0×

结论：  
> CUDA-L2 尤其擅长挖掘 **memory-bound / underutilized** 场景的隐藏性能空间。

---

## 5. Optimization Techniques Discovered by CUDA-L2

### 5.1 Abstraction Selection

* 小规模：raw WMMA（低开销）
* 大规模：CuTe abstractions（复杂 pipeline）

RL 自动权衡 **代码长度 vs 性能潜力**。

---

### 5.2 Non-divisible Tiling via Padding

* 允许 BM / BN **不整除维度**
* 通过 zero-padding 换取更优 tile shape
* 示例：
  * M=8192 → BM=160（padding 1.6%）
  * **+15.2% over cuBLASLt**

这是**人类工程师极少采用**的策略。

---

### 5.3 Advanced Kernel Techniques

CUDA-L2 熟练组合并参数化：

* Shared Memory Swizzle
* Multi-stage Pipelining
* Async Copy
* Register Accumulation
* Block Swizzle
* Epilogue Fusion

并进一步发现**变体策略**：

#### 5.3.1 Double-Buffered Register Fragments
* Ping-pong buffer 隐藏 latency

#### 5.3.2 Aggressive Multi-step Prefetch
* Prefetch 多个 K iteration

#### 5.3.3 Direct Register-to-Shared Epilogue
* 消除中间 tensor
* 使用 128-bit wide copy

#### 5.3.4 Staggered A/B Prefetch Scheduling
* A-prefetch → MMA → B-prefetch
* 提高 ILP

---

## 6. Analysis of Learned Policies

### 6.1 Tile Size Selection

* BM ∝ M，BN ∝ N（强相关）
* BK 与 K 相关性弱 → 受 register / pipeline 约束

### 6.2 Pipeline Stage Count

* K 小：2–3 stages
* K 大（>8K）：6+ stages

### 6.3 Block Swizzle Usage

* 小问题：可选
* 大问题：几乎必选（99%）
* stride 与 problem scale 强相关

---

## 7. Conclusion & Impact

CUDA-L2 首次在 **大规模 HGEMM 配置空间** 上系统性超越 **cuBLASLt-AutoTuning**，表明：

* **LLM-guided RL 可以超越人类专家 + 手工 heuristics**
* 即便是“已被榨干”的核心算子，仍存在结构性优化空间
* 自动化 kernel discovery 是未来 GPU 编程的重要方向

### Key Takeaways

* Contrastive RL 对性能优化至关重要  
* Execution + Profiling feedback > 静态规则  
* RL 能发现反直觉但有效的 kernel 设计  

---

*Reading date: 2025*
*Note status: Completed*
