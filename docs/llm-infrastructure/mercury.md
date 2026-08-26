# Mercury: Unlocking Multi-GPU Operator Optimization for LLMs via Remote Memory Scheduling

<div class="paper-meta" markdown>

**Authors**: Yue Guan, Xinwei Qiang, Zaifeng Pan, Daniels Johnson, Yuanwei Fang, Keren Zhou, Yuke Wang, Wanlu Li, Yufei Ding, Adnan Aziz  
**Institution**: UC San Diego; Meta; George Mason University; OpenAI; Rice University  
**Conference**: SOSP '25  
**Year**: 2025  
**Paper Link**: [ACM DOI](https://doi.org/10.1145/3731569.3764798)  
**GitHub**: [ChandlerGuan/mercury_artifact](https://github.com/ChandlerGuan/mercury_artifact)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">multi-gpu</span>
<span class="paper-tag">operator-compiler</span>
<span class="paper-tag">remote-memory</span>
</div>

## Background

Long-context attention and GEMM can exceed one GPU's HBM: the paper cites a 282 GB KV cache for Llama-3-70B at 32K context. Existing multi-GPU operators such as RingAttention and Ulysses are hand-designed for fixed communication patterns.

## Methodology

Mercury introduces CommIR, a loop-based IR that treats remote GPU memory as an explicitly schedulable level of the memory hierarchy. The compiler jointly searches tensor placement, loop partitioning, communication, and computation overlap rather than selecting from named parallelism templates.

| Abstraction | Role |
| --- | --- |
| Remote-memory load/store | Express cross-GPU data access directly |
| Loop transformations | Partition operator iteration spaces across GPUs |
| Schedule search | Explore placement, communication, and overlap choices |
| Code generation | Emit executable multi-GPU attention/GEMM kernels |

## Experiment

Mercury automatically reconstructs hand-optimized RingAttention and Ulysses schedules and finds configurations that outperform them in parts of the evaluated model/hardware space. The most important evidence is coverage: one compiler representation spans strategies previously implemented as separate systems.

## Limitation

- Search and cost-model accuracy become harder with network contention and dynamic workloads.
- Operator-level optimization does not choose a complete training/serving schedule.
- Generated strategies depend on supported CommIR transformations and collective primitives.

---

*Reading date: 2026-08*
*Note status: Completed*
