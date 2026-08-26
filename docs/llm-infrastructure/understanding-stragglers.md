# Understanding Stragglers in Large Model Training Using What-if Analysis

<div class="paper-meta" markdown>

**Authors**: Jinkun Lin, Ziheng Jiang, Zuquan Song, Sida Zhao, Menghan Yu, Zhanghan Wang, Chenyuan Wang, Zuocheng Shi, Xiang Shi, Wei Jia, Zherui Liu, Shuguang Wang, Haibin Lin, Xin Liu, Aurojit Panda, Jinyang Li  
**Institution**: New York University; ByteDance Seed; ByteDance; Zhejiang University  
**Conference**: OSDI '25  
**Year**: 2025  
**Paper Link**: [USENIX paper](https://www.usenix.org/conference/osdi25/presentation/lin-jinkun)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">stragglers</span>
<span class="paper-tag">llm-training</span>
<span class="paper-tag">what-if-analysis</span>
</div>

## Background

Synchronous LLM training amplifies small worker delays: pipeline stages and data-parallel groups repeatedly wait for the slowest participant. Traditional backup-worker or asynchronous-SGD techniques either consume too many resources or alter optimization semantics. The paper asks a more basic operational question first: how often do stragglers matter in real production training, and what causes them?

**Key Takeaways**

- A five-month ByteDance trace shows stragglers are common rather than exceptional hardware failures.
- What-if analysis reconstructs the dependency graph and simulates each operation at its non-straggling time.
- This yields causal “avoidable slowdown” estimates that ordinary aggregate utilization metrics cannot provide.

## Methodology

The analysis pipeline identifies operation dependencies across PP/DP/TP ranks, estimates a baseline execution time for each operation, and replays the job with selected stragglers removed. Comparing actual and counterfactual completion time isolates the impact of worker slowness without changing the training algorithm.

| Question | What-if comparison |
| --- | --- |
| Frequency | Actual jobs vs. jobs with all detected stragglers removed |
| Spatial pattern | Remove stragglers from selected workers or racks |
| Root cause | Partition slowdown by hardware, software, network, and workload factors |

## Experiment

| Finding | Result |
| --- | --- |
| Jobs materially affected | 42.5% of jobs are at least 10% slower because of stragglers |
| Tail waste | Slowdowns can waste 45% of allocated training time |
| Example scale | A 405B model run over 8K H100 GPUs shows the slowest GPU at 1.44x the computation latency of the fastest |
| Trace duration | Five months, January-May 2024 |

The important result is not a new mitigation algorithm but a measurement framework that turns a vague “cluster is slow” diagnosis into a ranked list of causal contributors. It also exposes temporal and spatial concentration, which can guide placement, maintenance, and admission policies.

## Limitation

- Counterfactual accuracy depends on the traced dependency model and the definition of a non-straggling baseline.
- What-if analysis diagnoses avoidable time but does not itself repair the root cause.
- Findings come from one production cluster and may not transfer to other network, GPU, or scheduler stacks.

---

*Reading date: 2026-08*
*Note status: Completed*
