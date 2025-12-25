# DuetServe: Harmonizing Prefill and Decode for LLM Serving via Adaptive GPU Multiplexing

<div class="paper-meta" markdown>

**Authors**: Lei Gao, Chaoyi Jiang, Hossein Entezari Zarch, Daniel Wong, Murali Annavaram  
**Institution**: University of Southern California, UC Riverside  
**Conference**: arXiv 2025  
**Paper Link**: [arXiv](https://arxiv.org/)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">LLM Serving</span>
<span class="paper-tag">GPU Multiplexing</span>
<span class="paper-tag">Prefill-Decode Isolation</span>
</div>

## Abstract

DuetServe 是一个统一的 LLM serving 框架，能够在单个 GPU 内实现 disaggregation 级别的隔离。系统默认运行在 aggregated 模式，当预测到 TBT (Time-Between-Tokens) 可能降级时，动态激活 SM 级别的 GPU 空间复用。

**Key Contributions**:
- 提出 attention-aware roofline model 预测迭代延迟，提前检测潜在 TBT 违规
- 动态 GPU 分区优化器，确定 prefill 和 decode 之间的最优 SM 分配
- 无中断执行引擎，通过 look-ahead decode scheduling 和异步 kernel dispatch 消除 CPU-GPU 同步开销
- 相比 SOTA 框架，在保持低 TBT 延迟的同时实现最高 1.3x 的吞吐量提升

## Core Ideas

现有 LLM serving 方法面临两难：aggregated 方式会导致 prefill 和 decode 之间的干扰，降低 TBT；disaggregated 方式虽然改善延迟，但因模型复制和 KV cache 传输而浪费资源。

DuetServe 的核心思想是通过细粒度的 SM 级分区，在单 GPU 内按需解耦 prefill 和 decode 执行。系统将 GPU SMs 划分为逻辑上独立的"子 GPU"，仅在干扰威胁延迟 SLO 时隔离 prefill 和 decode 工作负载，否则恢复到完全共享执行。

---

*Reading date: 2025*
*Note status: Completed*
