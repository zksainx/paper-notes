# BatchGen: An Architecture for Scalable and Efficient Batch Inference

<div class="paper-meta" markdown>

**Authors**: Tairan Xu, Leyang Xue, Zhan Lu, Jinfu Deng, Hongyang Xiao, Yinsicheng Jiang, Congjie He, Matej Sandor, Le Xu, Luo Mai  
**Institution**: University of Edinburgh; Tencent  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/xu-tairan)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">batch-inference</span>
<span class="paper-tag">coroutines</span>
<span class="paper-tag">moe</span>
</div>

## Background

Offline inference optimizes batch completion time, not per-request latency. Sparse experts receive tiny batches and heavy-tailed reasoning sequences leave fleet-wide stragglers, but serving engines pin each sequence/state to one GPU.

## Methodology

BatchGen represents every sequence as an event-driven coroutine with `yield`, `combine`, `partition`, and `migrate`. Module boundaries become scheduling points for forming larger expert batches and continuously redistributing long-tail work.

## Experiment

On 128 GPUs, BatchGen reduces batch completion time by up to 2.3x; on memory-constrained accelerators it outperforms the strongest offloading baseline by up to 9.6x.

## Limitation

- Coroutine/state migration adds metadata and network traffic.
- Fine-grained scheduling favors throughput over interactive latency.
- Dynamic reorganization complicates deterministic execution/debugging.

---

*Reading date: 2026-08*
*Note status: Completed*
