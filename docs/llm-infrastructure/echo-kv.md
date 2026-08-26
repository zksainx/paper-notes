# ECHO: Efficient KV Cache Offloading with Lossless Prefetching for Native Sparse Attention LLMs

<div class="paper-meta" markdown>

**Authors**: Guangda Liu, Wenhao Chen, Chengwei Li, Zhenyu Ning, Jing Lin, Yiwu Yao, Quan Chen, Shixuan Sun, Jieru Zhao, Minyi Guo  
**Institution**: Shanghai Jiao Tong University; Huawei; Guizhou University  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/liu-guangda)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">sparse-attention</span>
<span class="paper-tag">kv-offload</span>
<span class="paper-tag">prefetch</span>
</div>

## Background

Native sparse attention reduces attention compute but does not reduce KV capacity, limiting concurrency for 100K-token requests. Naive host offload makes recall the bottleneck.

## Methodology

ECHO uses a GPU-graph-compatible cache manager, predicts token-index selection from numerical score stability, prefetches within and across queries, and fuses recall with indexer computation in a pipelined GPU kernel.

## Experiment

ECHO improves long-context generation throughput by up to 2.1x over SGLang/vLLM while retaining comparable light-load latency. Inter-query prefetch alone reaches about 1.1x in the evaluated random workload; most gain comes from intra-query prediction and fused overlap.

## Limitation

- Requires native sparse-attention models exposing early selection signals.
- Offload remains a throughput/latency trade-off under light load.
- Prediction benefit depends on score and query locality.

---

*Reading date: 2026-08*
*Note status: Completed*
