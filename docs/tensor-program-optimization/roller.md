# ROLLER: Fast and Efficient Tensor Compilation for Deep Learning

<div class="paper-meta" markdown>

**Authors**: Hongyu Zhu, Ruofan Wu, Yijia Diao, Shanbin Ke, Haoyu Li, Chen Zhang, Jilong Xue, Lingxiao Ma, Yuqing Xia, Wei Cui, Fan Yang, Mao Yang, Lidong Zhou, Asaf Cidon, Gennady Pekhimenko  
**Institution**: Microsoft Research  
**Conference**: OSDI '22 (16th USENIX Symposium on Operating Systems Design and Implementation)  
**Year**: 2022  
**Paper Link**: [USENIX OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/zhu)  
**GitHub**: [microsoft/nnfusion](https://github.com/microsoft/nnfusion/tree/osdi22_artifact/artifacts)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Tensor Compilation</span>
<span class="paper-tag">Deep Learning Compiler</span>
<span class="paper-tag">Performance Optimization</span>
<span class="paper-tag">Construction-Based Approach</span>
<span class="paper-tag">rTile</span>
<span class="paper-tag">NNFusion</span>
</div>

## Abstract

ROLLER 是一个快速高效的深度学习张量编译器，采用了创新的 **construction-based（基于构造）** 方法来生成高性能的算子 kernel。与传统的 search-based（基于搜索）张量编译器（如 TVM、Ansor）需要数小时甚至数天的编译时间不同，ROLLER 能够在**数秒内**生成高效的 kernel 代码，同时在主流加速器（GPU）上达到与最先进方案相当的性能，在新兴加速器（如 Graphcore IPU）上甚至表现更优。

ROLLER 的核心是 **rTile** 抽象，这是一种新的 tile 抽象机制，能够封装与底层加速器关键特性（内存 bank、内存事务长度、最小可调度单元）对齐的张量形状。通过递归的 rTile 构造算法和微性能模型（micro-performance model），ROLLER 将张量编译问题从无约束的组合优化转变为高效的"白盒"构造过程。

**Key Contributions**:

- 提出了一种 construction-based 的张量编译方法，将编译时间从小时级降至秒级（平均 0.24 秒 vs Ansor 的 0.99 小时）
- 设计了 rTile 抽象，通过与硬件特性对齐来限制搜索空间并提升执行效率
- 实现了硬件抽象层（HAL）和微性能模型，支持快速评估程序性能而无需在真实设备上运行
- 在 100+ 主流 DNN 算子上验证：54.6% 的 kernel 优于 TVM，58.8% 优于 Ansor，59.7% 优于 NVIDIA 库
- 展示了跨平台能力：在 NVIDIA GPU、AMD GPU 和 Graphcore IPU 上均表现出色

## Core Ideas

ROLLER 从根本上改变了张量编译的范式。传统的 search-based 编译器将张量编译视为嵌套循环优化的组合优化问题，需要通过机器学习算法在巨大的搜索空间中探索数小时才能找到高性能的配置。ROLLER 则采用 construction-based 方法，通过引入 rTile 抽象和硬件对齐约束，直接构造出高性能的张量程序。

rTile 是 ROLLER 的核心创新。它封装了与加速器硬件特性严格对齐的张量形状，这些特性包括内存 bank 数量、内存事务长度和最小可调度单元。通过强制对齐，可用的 tile 形状被大幅限制，使得编译器只需构造一个能够饱和执行单元的对齐 tile 形状即可最大化流水线吞吐量。这种"白盒"构造过程从本质上比求解原始的无约束组合优化问题更高效，显著缩短了编译时间。

ROLLER 采用递归算法构造基于 rTile 的程序（rProgram），并通过微性能模型快速评估其性能，无需在真实设备上执行。系统将 DNN 算子中的计算视为数据处理流水线，数据 tile 在具有多层内存层次结构的抽象硬件中移动和处理。通过硬件抽象层，ROLLER 能够静态推导出 rTile 的内存占用（包括 padding）和跨不同内存层的内存流量，从而实现快速性能预测。

---

## 1. Introduction & Motivation

### 1.1 传统 Tensor Compilation 的挑战

深度学习编译器的核心任务是将高层神经网络算子转换为在特定硬件加速器上高效执行的低层 kernel 代码。传统的张量编译器（如 TVM、Ansor）采用 **search-based** 方法：

* **问题建模**：将张量编译视为嵌套循环优化的组合优化问题
* **搜索空间**：由于硬件架构的复杂性和优化选项的组合爆炸，搜索空间极其庞大
* **编译时间**：需要使用机器学习算法（如进化算法、强化学习）探索数小时甚至数天
* **性能瓶颈**：
  - Ansor 编译单个算子平均需要 0.65 小时
  - 一个 ResNet 卷积算子的调优可能需要 2.17 小时
  - 端到端模型调优可能需要数周时间

### 1.2 ROLLER 的解决方案

ROLLER 提出了一种全新的 **construction-based** 范式：

* **核心思想**：与其在巨大的搜索空间中搜索，不如通过硬件对齐约束直接构造高性能程序
* **关键洞察**：加速器要高效工作，数据 tile 的形状必须与硬件特性对齐（内存 bank、事务长度、调度单元）
* **性能优势**：
  - 编译速度：数秒级（平均 0.24 秒）
  - 性能质量：与 SOTA 相当或更优
  - 跨平台：在成熟的 GPU 和新兴的 IPU 上均表现良好

---

## 2. Technical Approach

### 2.1 Construction-Based vs Search-Based

**Search-Based 方法（TVM, Ansor）**：

* 将问题建模为组合优化：寻找最优的循环变换、tiling 策略、内存映射等
* 使用搜索算法（遗传算法、模拟退火、强化学习）探索配置空间
* 需要大量性能测量或代价模型评估
* 优点：理论上能找到非常优化的配置
* 缺点：**编译时间长**（小时到天级别），对于大型或定制算子尤其慢

**Construction-Based 方法（ROLLER）**：

* 通过硬件对齐约束限制设计空间
* 递归构造满足约束的 tile 形状和程序结构
* 使用微性能模型快速评估，无需真实执行
* 优点：**编译速度快**（秒级），适合快速迭代和部署
* 缺点：在小算子上可能比搜索方法慢 50%，FLOPS 可达搜索方法的 70%

**关键权衡**：

* ROLLER 牺牲了对小算子的极致性能优化，换取了数量级的编译速度提升
* 对于大型、昂贵的定制算子，ROLLER 的快速编译能力尤其有价值
* 可以结合两种方法：先用 ROLLER 快速生成基线，再用搜索方法精调

### 2.2 rTile Abstraction

**rTile** 是 ROLLER 的核心抽象，代表与硬件特性对齐的张量 tile。

**定义**：rTile 封装以下信息：

* **Tensor Expression (expr)**：tile 对应的张量表达式
* **Shape**：tile 的形状，必须与硬件对齐
* **Alignment Constraints**：与硬件特性的对齐要求
  - Memory bank alignment（内存 bank 对齐）
  - Transaction length alignment（事务长度对齐）
  - Minimum schedulable unit alignment（最小调度单元对齐）

**对齐的重要性**：

* **内存 bank**：GPU 共享内存通常分为多个 bank（如 32 个），访问同一 bank 会导致 bank conflict
* **事务长度**：全局内存访问以固定长度的事务为单位（如 32 字节），对齐可提高内存带宽利用率
* **调度单元**：GPU 以 warp（32 线程）为单位调度，tile 大小应是 warp 大小的倍数

**rTile 的优势**：

* 通过强制对齐，大幅减少可行的 tile 形状数量
* 只需找到一个**饱和执行单元**的对齐 tile，即可达到接近最优的吞吐量
* 将搜索问题转化为构造问题，使问题更易处理

### 2.3 Hardware Abstraction Layer (HAL)

ROLLER 通过 HAL 抽象不同加速器的硬件特性，使构造算法具有可移植性。

**内存层次结构**：

* **L2 层**：全局内存 + L2 缓存
* **L1 层**：共享内存（GPU）或片上内存
* **L0 层**：寄存器

**硬件参数（以 NVIDIA V100 GPU 为例）**：

* **全局内存事务长度**：32 字节（8 个 float 元素）
* **共享内存 bank**：32 个，每个 bank 长度 4 字节
* **Warp 大小**：32 线程
* **内存带宽**：通过 micro-benchmark 测量各层带宽

**HAL 的作用**：

* 提供统一的硬件特性描述接口
* 支持多种加速器：NVIDIA GPU、AMD GPU、Graphcore IPU
* 使构造算法能够根据硬件特性自动调整

**micro-benchmark**：

* 在目标硬件上运行小型基准测试，测量实际内存带宽
* 比理论峰值带宽更准确，考虑了缓存、预取等实际因素

---

## 3. System Architecture

### 3.1 Recursive Construction Algorithm

ROLLER 采用递归的树结构算法来构造基于 rTile 的程序（rProgram）。

**核心思想**：

* 将 DNN 算子的计算视为**数据处理流水线**
* 数据 tile 在多层内存层次中移动：L2 → L1 → L0 → 计算 → L0 → L1 → L2
* 目标是最大化流水线的吞吐量

**构造过程**：

1. **选择 rTile 形状**：从符合硬件对齐约束的候选形状中选择
2. **递归分解**：将大的 rTile 递归分解为更小的 sub-rTile
3. **内存分配**：确定每个 rTile 在内存层次中的位置（L2/L1/L0）
4. **性能评估**：使用微性能模型评估当前 rProgram 的性能
5. **迭代优化**：尝试不同的 rTile 形状和分解策略

**树结构**：

* 树的每个节点代表一个 rTile
* 父节点表示更大的 tile，子节点表示分解后的 sub-tile
* 叶节点表示在寄存器（L0）中处理的最小 tile

**对齐约束的应用**：

* 每一层分解都必须满足对齐要求
* 显著减少了需要考虑的分解方案数量
* 使得构造过程能够快速收敛

### 3.2 Micro-Performance Model

微性能模型是 ROLLER 实现快速编译的关键，它能够**无需在真实设备上执行**即可评估 rProgram 的性能。

**性能指标**：

* **Pipeline Throughput（流水线吞吐量）**：单位时间内处理的数据量
* **计算方式**：$Throughput = \frac{Computation}{max(T_{compute}, T_{memory})}$

**性能建模步骤**：

**1. 计算量分析**：
- 从 tensor expression 推导出浮点运算次数（FLOPs）

**2. 内存流量分析**：
- 根据 rTile 形状和内存层次，计算每层的内存读写量
- 包括 padding 导致的额外开销
- 公式：从 rTile 的形状和数据访问模式静态推导

**3. 执行时间估算**：
- **计算时间**：$T_{compute} = \frac{FLOPs}{Peak\_FLOPS}$
- **内存时间**：$T_{memory} = \sum_{layer} \frac{Traffic_{layer}}{Bandwidth_{layer}}$
- 实际时间取两者的最大值（计算密集 vs 内存密集）

**4. 硬件资源饱和度**：
- 检查 rTile 是否能够饱和执行单元（SM/CU）
- 检查是否存在资源瓶颈（寄存器、共享内存）

**模型优势**：

* **速度快**：纯静态分析，毫秒级完成
* **准确度**：与真实性能高度相关，能够区分好坏配置
* **可扩展**：容易适配新的硬件架构

**模型局限**：

* 不能精确预测绝对性能（如 bank conflict 的细节影响）
* 主要用于**相对排序**，选择最佳的 rProgram 候选

### 3.3 Pipeline View of Computation

ROLLER 采用独特的**流水线视角**来理解和优化张量计算。

**传统视角 vs ROLLER 视角**：

* **传统**：循环嵌套 + 循环变换（tiling, fusion, reordering）
* **ROLLER**：数据流水线 + tile 移动和处理

**流水线阶段**：

1. **Load from L2 to L1**：从全局内存加载 tile 到共享内存
2. **Load from L1 to L0**：从共享内存加载到寄存器
3. **Compute**：在寄存器上执行计算
4. **Store from L0 to L1**：结果写回共享内存
5. **Store from L1 to L2**：结果写回全局内存

**优化目标**：

* 最大化流水线吞吐量
* 平衡计算和内存访问（达到 compute-bound 或 memory-bound 的理论极限）
* 减少内存层次间的数据搬运

**优化策略**：

* **Tile shape 选择**：选择能最大化重用的形状
* **Fusion**：将多个算子融合到同一流水线，减少内存往返
* **Padding**：通过 padding 达到对齐要求，提升内存效率

---

## 4. Experimental Results

### 4.1 Performance Comparison

ROLLER 在超过 100 个主流 DNN 模型的算子上进行了全面评估。

**测试平台**：

* NVIDIA V100 GPU
* AMD MI50 GPU
* Graphcore IPU

**与 DNN 库的比较（NVIDIA V100）**：

* **vs NVIDIA cuDNN/cuBLAS**：59.7% 的 kernel 性能更优
* **vs AMD ROCm 库**：73.1% 的 kernel 性能更优

**与 Tensor 编译器的比较**：

* **vs TVM**：54.6% 的 kernel 性能更优
* **vs Ansor**：58.8% 的 kernel 性能更优（Ansor 是 OSDI'20 的 SOTA 工作）

**性能分布**：

* 在大型、复杂算子上，ROLLER 表现尤其突出
* 在小算子上，ROLLER 平均比 Ansor 慢约 50%
  - 原因：小算子的优化空间有限，搜索方法能找到极致配置
  - 但编译时间差异巨大：ROLLER 秒级 vs Ansor 小时级

**跨平台性能**：

**AMD GPU**：ROLLER 在 AMD 上表现更好（73.1% 优于 AMD 库）
- 原因：AMD 的软件栈不如 NVIDIA 成熟，ROLLER 能填补这一空白

**Graphcore IPU**：ROLLER 提供了更好的 kernel
- 原因：IPU 是新兴架构，缺乏成熟的库，ROLLER 的快速适配能力优势明显

### 4.2 Compilation Efficiency

ROLLER 最大的优势在于**编译速度**的数量级提升。

**编译时间对比（单个算子）**：

* **ROLLER**：平均 0.24 秒，中位数约 0.1 秒
* **Ansor**：平均 0.99 小时（3564 秒），中位数约 0.65 小时
* **加速比**：约 **14850x** 的编译速度提升

**具体案例**：

**ResNet 卷积层**：
- Ansor：2.17 小时
- ROLLER：< 1 秒

**BERT Transformer**：
- Ansor：端到端模型调优需要数天
- ROLLER：所有算子编译在数分钟内完成

**实际影响**：

* **快速迭代**：研究人员可以快速尝试新的网络架构
* **部署便利**：在生产环境中可以实时为新模型生成优化 kernel
* **定制算子**：对于用户定义的复杂算子，不再需要等待漫长的调优

**能量效率**：

* 编译过程能耗降低数千倍
* 对于云端编译服务，成本显著降低

### 4.3 Cross-Platform Results

ROLLER 的硬件抽象层使其能够轻松适配不同的加速器架构。

**支持的平台**：

**1. NVIDIA GPU**（V100, A100）
   - HAL 配置：L2（全局内存+缓存），L1（共享内存），L0（寄存器）
   - 对齐参数：32 字节事务，32 bank 共享内存，32 线程 warp

**2. AMD GPU**（MI50, MI100）
   - 类似的内存层次，但 bank 配置和 warp 大小可能不同
   - ROLLER 通过 micro-benchmark 自动检测参数

**3. Graphcore IPU**
   - 不同的内存架构：片上 SRAM，无传统缓存层次
   - ROLLER 通过 HAL 适配其特殊的 tile 结构

**移植性验证**：

* 在 NVIDIA GPU 上训练/调优的构造算法，可以直接应用到 AMD GPU 和 IPU
* 只需更新 HAL 的硬件参数，无需修改构造逻辑
* 在新硬件上达到 SOTA 性能，编译时间仍保持秒级

**新硬件适配流程**：

1. 运行 micro-benchmark 获取内存带宽
2. 配置 HAL 参数（bank 数量、事务长度、调度单元大小）
3. 运行 ROLLER 构造算法
4. （可选）微调构造策略

**启示**：

* Construction-based 方法的可移植性优于 search-based
* Search-based 方法在新硬件上可能需要重新搜索数天
* ROLLER 的白盒构造直接利用硬件特性，适配更快

---

## 5. Limitations & Future Work

### 5.1 当前局限性

**性能局限**：

* 在**小算子**上比 Ansor 慢约 50%
  - 小算子优化空间有限，搜索方法能找到更精细的配置
  - 但编译时间差异巨大（秒 vs 小时）
* FLOPS 可能仅达搜索方法的 **70%**
  - 树结构的单目标优化限制了探索多样性
  - 未来可考虑混合方法：ROLLER 生成初始配置 + 局部搜索精调

**算法局限**：

* **单目标优化**：当前递归树只优化吞吐量，未考虑能耗、延迟等多目标
* **贪心构造**：递归构造是贪心的，可能错过全局最优解
* **Fusion 受限**：算子融合策略相对简单，不如搜索方法全面

**硬件覆盖**：

* 主要针对 GPU 架构优化，对 CPU、TPU 等支持有限
* 对于特殊架构（如稀疏加速器），需要扩展 HAL 和构造算法

### 5.2 混合方法的潜力

**ROLLER + Search 的协同**：

* **第一阶段**：用 ROLLER 快速生成高质量 baseline（秒级）
* **第二阶段**：用搜索方法在 baseline 附近局部精调（分钟到小时级）
* **优势**：
  - 避免从零开始搜索的巨大开销
  - 保留搜索方法的精调能力
  - 总编译时间显著缩短

**针对小算子的策略**：

* 维护一个小算子的搜索结果缓存
* 新模型中的小算子直接查表
* 只对大型或定制算子使用 ROLLER 快速编译

### 5.3 未来研究方向

**多目标优化**：

* 扩展微性能模型，同时优化吞吐量、能耗、延迟
* 使用 Pareto 前沿选择不同场景的最优配置

**自适应构造**：

* 根据算子特征（计算密集 vs 内存密集）动态调整构造策略
* 使用机器学习预测最佳构造路径

**更广泛的硬件支持**：

* 扩展到 TPU、稀疏加速器、FPGA
* 研究 ROLLER 在异构系统中的应用

**算子融合增强**：

* 更智能的 fusion 策略，考虑跨算子的 rTile 重用
* 支持更复杂的计算图模式

---

## 6. Conclusion & Impact

ROLLER 代表了张量编译领域的一次重要范式转变，从 search-based 转向 construction-based。通过引入 rTile 抽象和硬件对齐约束，ROLLER 将编译时间从小时级降至秒级，同时在多数情况下保持了与 SOTA 相当的性能。这一突破对深度学习系统的研究和实际部署都具有深远影响。

### Key Insights

* **硬件对齐是关键**：强制对齐约束能够大幅缩小搜索空间，将组合优化转化为构造问题
* **白盒优于黑盒**：利用硬件特性的白盒构造比盲目搜索更高效
* **微性能模型的威力**：快速准确的性能预测模型使得无需真实执行即可评估方案
* **编译速度的价值**：秒级编译使得快速迭代、实时部署和定制算子优化成为可能
* **可移植性优势**：Construction-based 方法更容易适配新硬件，适应快速演进的 AI 加速器生态

### Impact on DL Systems

**对研究的影响**：

* 降低了张量编译的实验门槛，研究人员可以快速验证想法
* 为新架构和新算子的快速原型开发提供了工具

**对工业部署的影响**：

* 使得实时编译和动态优化成为可能
* 降低了维护多硬件平台优化 kernel 的成本
* 加速了新模型从研究到生产的转化

**对硬件生态的影响**：

* 为新兴加速器（如 IPU）提供了快速的软件栈支持
* 降低了新硬件的软件生态建设难度

### Future Directions

* 探索 ROLLER 与搜索方法的混合策略，结合两者优势
* 扩展到更多硬件平台（TPU、稀疏加速器、边缘设备）
* 研究多目标优化（性能、能耗、延迟）的构造算法
* 将 ROLLER 集成到端到端深度学习框架（如 PyTorch、TensorFlow）

---

*Reading date: 2025*
*Note status: Completed*
