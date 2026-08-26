# WLB-LLM: Workload-Balanced 4D Parallelism for Large Language Model Training

<div class="paper-meta" markdown>

**Authors**: Zheng Wang, Anna Cai, Xinfeng Xie, Zaifeng Pan, Yue Guan, Weiwei Chu, Jie Wang, Shikai Li, Jianyu Huang, Chris Cai, Yuchen Hao, Yufei Ding  
**Institution**: UC San Diego; Meta  
**Conference**: OSDI '25  
**Year**: 2025  
**Paper Link**: [USENIX paper](https://www.usenix.org/conference/osdi25/presentation/wang-zheng)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">llm-training</span>
<span class="paper-tag">parallelism</span>
<span class="paper-tag">load-balancing</span>
</div>

## Background

Modern LLM training combines data, pipeline, context, and tensor parallelism. Equal token counts do not imply equal work because masked attention makes later tokens in long documents more expensive. In an 8K-GPU 405B training job with 128K context, the slowest GPU can take 1.44x the computation time of the fastest, forcing synchronized workers to wait.

**Key Takeaways**

- The imbalance has two distinct locations: micro-batches at pipeline parallelism and document shards at context parallelism.
- Variable-length packing balances real attention work rather than token count.
- Per-document context sharding gives each worker in a CP group an equal workload while preserving training semantics.

## Methodology

| Level | Technique | Purpose |
| --- | --- | --- |
| Pipeline parallelism | Workload-aware variable-length packing | Equalize micro-batch compute/communication |
| Pipeline scheduling | Delay extreme outlier documents | Prevent one long document from setting the step critical path |
| Context parallelism | Fine-grained per-document sharding | Equalize work within each CP group |
| Training semantics | Randomized data-loader compatible heuristics | Avoid convergence changes from deterministic reordering |

The design follows the critical-path structure of 4D training: PP latency is controlled by the heaviest micro-batch, while CP latency is controlled by the heaviest worker shard. Optimizing these independently avoids adding a global repartitioning barrier.

## Experiment

| Result | Evidence |
| --- | --- |
| Overall speedup | Average 1.23x in an internal LLM training framework |
| Context scaling | Speedup rises from about 1.15x at shorter contexts to 1.30x at 128K in the reported study |
| Component breakdown | CP sharding contributes about 1.02x; outlier-document delay and packing provide the larger gain |
| Convergence | 550M-model loss curve closely tracks the fixed-length baseline |

The paper also reports that naive packing can hurt kernel efficiency. WLB-LLM therefore optimizes the imbalance metric subject to a bounded packing overhead rather than seeking perfect arithmetic balance at any cost.

## Limitation

- Heuristics depend on document-length distributions and attention-mask structure.
- Very irregular MoE routing can introduce a second imbalance source not solved by document packing.
- The implementation is integrated into an internal training framework, so portability details are limited.

---

*Reading date: 2026-08*
*Note status: Completed*
