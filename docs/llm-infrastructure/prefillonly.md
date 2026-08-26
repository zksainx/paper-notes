# PrefillOnly: An Inference Engine for Prefill-Only Workloads in Large Language Model Applications

<div class="paper-meta" markdown>

**Authors**: Kuntai Du, Bowen Wang, Chen Zhang, Yiming Cheng, Qing Lan, Hejian Sang, Yihua Cheng, Jiayi Yao, Xiaoxuan Liu, Yifan Qiao, Ion Stoica, Junchen Jiang  
**Institution**: University of Chicago; Tsinghua University; LinkedIn; UC Berkeley  
**Conference**: SOSP '25  
**Year**: 2025  
**Paper Link**: [arXiv:2505.07203](https://arxiv.org/abs/2505.07203)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">prefill</span>
<span class="paper-tag">jct-scheduling</span>
<span class="paper-tag">kv-cache</span>
</div>

## Background

Recommendation, credit verification, and labeling often ask an LLM for exactly one output token. General engines retain KV for future decode and schedule under unknown output length, wasting memory and ignoring deterministic job completion time (JCT).

## Methodology

| Technique | Effect |
| --- | --- |
| Hybrid prefilling | Retain only cache needed by the active/last layer and discard non-reusable suffix KV |
| Predictable JCT | Estimate work from uncached input tokens; correlation reaches 0.987 in one Qwen-32B setup |
| Continuous calibration | Recompute waiting-job JCT as prefix-cache residency changes |
| Aging | Offset JCT by waiting time to avoid starvation |

Because prefill is compute-bound, batching does not yield decode-style throughput gains and can inflate latency. PrefillOnly therefore schedules smaller/cache-hit requests using continuously calibrated SRJF.

## Experiment

| Setting | Result |
| --- | --- |
| Hardware/models/traces | Four hardware setups, three LLMs, two simulated application traces |
| Supported arrival rate | 1.4x-4.0x higher QPS without increasing average or P99 latency |
| Chunked prefill motivation | 14% throughput loss for a 20K-token input with 512-token chunks in the reported measurement |

## Limitation

- The design applies only when exactly one or a known small number of output tokens is required.
- JCT profiles must be regenerated for model/hardware changes.
- Aggressive KV dropping reduces opportunities if a supposedly prefill-only request later becomes generative.

---

*Reading date: 2026-08*
*Note status: Completed*
