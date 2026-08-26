# Revisiting Pipeline Parallelism for LLM Serving

<div class="paper-meta" markdown>

**Authors**: Soonjae Hwang, Jeongseob Ahn  
**Institution**: Korea University  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/hwang)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">pipeline-parallelism</span>
<span class="paper-tag">chunked-prefill</span>
<span class="paper-tag">delay-scheduling</span>
</div>

## Background

Tensor parallelism has low latency with NVLink but high communication on PCIe. Pipeline parallelism communicates less yet online variation in arrivals, prompt lengths, and prefill/decode mixtures creates bubbles.

## Methodology

Greedy and predictive controllers resize prefill chunks using TTFT/TPOT slack; delay scheduling redistributes decode requests among stages. The predictive variant models linear and attention latency components.

## Experiment

On four A100 40GB GPUs serving Qwen2.5-14B/32B, optimized PP outperforms TP. Dynamic chunking yields up to 1.4x goodput over static PP; delay scheduling lowers P99 TPOT/end-to-end latency by about 25% in a reported workload.

## Limitation

- Pipeline stages duplicate/partition state and require balanced layer placement.
- Prediction errors and burstiness can violate slack assumptions.
- TP remains preferable on strong interconnects and tight single-request latency.

---

*Reading date: 2026-08*
*Note status: Completed*
