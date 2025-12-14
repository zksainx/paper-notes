# GPU Performance Portability needs Autotuning

<div class="paper-meta" markdown>

**Authors**: Burkhard Ringlein, Thomas Parnell, Radu Stoica  
**Institution**: IBM Research Europe  
**Conference**: arXiv 2025  
**Paper Link**: [arXiv:2505.03780](https://arxiv.org/abs/2505.03780)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">GPU Portability</span>
<span class="paper-tag">Autotuning</span>
<span class="paper-tag">LLM Inference</span>
</div>

## Abstract

This paper makes the case for combining just-in-time (JIT) compilation with comprehensive kernel parameter autotuning to enable portable LLM inference with state-of-the-art performance without code changes, addressing vendor lock-in and hardware portability challenges.


## Q1: Can Autotuning enable portable SOTA performance?
**(自动调优能否实现可移植的 SOTA 性能?)**

**结论: 是的. 自动调优的 Triton 内核在极低的代码量下, 能在不同平台上展现出广泛的竞争力.**

* **总体效率**: 代码量(LoC)不到 SOTA 实现的 2%.
* **性能上限**: 在最佳情况下, 比 SOTA(flash_attn)快 **2.3倍**.
* **性能下限**: 在最差情况下(无任何手动优化), 仍能达到 SOTA 性能的 **78%**.
* **平台表现**:
    * **AMD MI250**: 自动调优内核始终优于通过 hipify 交叉编译的 CUDA 代码(平均快 **20%** 以上).
    * **NVIDIA A100**: 在大多数场景下达到 SOTA 基线的 **91%-98%**.(小负载下的性能差距归因于 Triton 编译器未充分利用 FP16 优化, 而非调优参数问题).

---

## Q2: Is autotuning necessary?
**(自动调优是否必要?)**

**结论: 绝对必要. 简单的配置复用行不通, 自动调优是挖掘编译器潜力的关键.**

* **配置不可复用性 (Non-portability)**:
    * 将一个 GPU 上的最优配置直接用于另一个 GPU, 会导致严重的性能下降(仅剩原性能的 **7%**, 即下降一个数量级).
    * 部分配置在异构平台上甚至是无效的(invalid).
* **代码生成优势**:
    * **深度优化**: 自动调优不仅仅是选择参数, 它能触发 JIT 编译器进行手动模板无法实现的优化(如循环展开, 软件流水线).
    * **指令多样性**: 分析显示, 自动调优生成的汇编代码比手动 CUDA 模板更复杂, 指令更丰富(唯一指令数 **475 vs 224**), 表明其探索了更广阔的解空间.

---

## Q3: Why is Triton autotuning not used in practice?
**(为什么 Triton 自动调优在实践中未被广泛采用?)**

**结论: 尽管有效, 但现有的实现存在工程和生态上的阻碍. 目前主流框架(如 vLLM, sglang)很少使用它.**

主要原因有三点:

1.  **运行时开销大**:
    * 内置自动调优器需要按顺序尝试所有配置, 每次都需要 JIT 编译和执行, 开销巨大.
    * 调优结果无法跨进程缓存, 每次重启进程都需要重新调优.
2.  **社区优先级不同**:
    * 工业界和研究界的首要目标通常是在"事实标准平台"(NVIDIA)上跑出高性能, 跨平台移植性是次要目标.
    * 代码成熟度不足.
3.  **仍需人工干预**:
    * 自动调优仍需开发者手动提供"配置搜索列表".
    * 依赖程序员直觉定义的搜索空间往往不是最优的, 可能导致近 **20%** 的性能差距.

---

## Q4: What are the gaps towards practical autotuning?
**(实现实用化自动调优还存在哪些差距?)**

**结论: 为了让自动调优真正落地, 需要解决以下四个技术缺口:**

1.  **更好的 API**: 开发者需要高层级的 API 来更方便地定义内核参数配置空间及其依赖关系.
2.  **高效的搜索算法**: 目前的搜索空间巨大(每种 Tensor 形状可达 1000 种配置), 需要先进的搜索方法来替代暴力搜索, 以减少调优时间.
3.  **可复用的缓存机制**: 调优结果必须能够持久化缓存, 并包含环境依赖信息, 以避免重复调优.
4.  **将调优移出关键路径**:
    * **离线调优 (Ahead-of-time)**: 作为开发流程的一部分提前完成.
    * **闲时调优**: 利用 GPU 空闲时间根据工作负载指标进行调优.

---

*Reading date: 2025*
*Note status: Completed*
