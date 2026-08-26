# Achieving Cloud-Grade SLOs for Local MoE Inference through CPU-GPU Hybrid Design

<div class="paper-meta" markdown>

**Authors**: Wenxin Wang, Yule Hou, Yu Ji, Peng Qu, Youhui Zhang  
**Institution**: Tsinghua University; Xingyun Integrated Circuits  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/wang-wenxin)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">moe-inference</span>
<span class="paper-tag">local-serving</span>
<span class="paper-tag">cpu-gpu-hybrid</span>
</div>

## Background

Local MoE systems often require quantization/rerouting, miss a 30-second long-prefill TTFT, decode below 20 tokens/s, and lose concurrency. The paper targets intact DeepSeek-V3-class models on commodity CPUs and consumer GPUs.

## Methodology

Stream-loading prefill and distributed SLP overlap CPU expert weights with GPU work; zero-copy shared weights enable intra-node prefill/decode disaggregation; dual-batch scheduling overlaps attention and MoE; AVX-512 FP8 GEMV and fine-grained CPU parallelism accelerate routed experts.

## Experiment

Single/two-GPU prefill reaches 1,200/1,800 tokens/s and handles 32K/45K prompts within 30 s. Decode reaches 28 tokens/s for INT4 and 21.5 tokens/s for intact FP8 DeepSeek-V3; mixed concurrency gains 50% throughput with under 15% latency increase.

## Limitation

- Requires high-end dual-socket CPUs, large DRAM, and recent consumer GPUs.
- AVX-512/FP8 tuning is platform-specific.
- “Cloud-grade” is defined by selected SLOs, not cloud fleet availability.

---

*Reading date: 2026-08*
*Note status: Completed*
