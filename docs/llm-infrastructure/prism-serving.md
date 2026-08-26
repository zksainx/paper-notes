# Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning

<div class="paper-meta" markdown>

**Authors**: Shan Yu, Yifan Qiao, Mingyuan Ma, Yangmin Li, Shuo Yang, Xinyuan Tong, et al.  
**Institution**: UCLA; UC Berkeley; Harvard; CMU; industry collaborators  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/yu-shan)  
**GitHub**: [ovg-project/kvcached](https://github.com/ovg-project/kvcached)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">multi-model-serving</span>
<span class="paper-tag">memory-ballooning</span>
<span class="paper-tag">production</span>
</div>

## Background

Production traces show shifting bursty groups of simultaneously active models. Static space sharing strands memory during idle periods; time sharing pays activation latency during rapid bursts. Prism treats elastic GPU memory as the common mechanism behind both.

## Methodology

The `kvcached` balloon driver reclaims and restores model/KV memory, while a global scheduler coordinates oversubscription, activation, and request placement across models. This lets hot models expand and cold models shrink without selecting one fixed sharing mode.

## Experiment

| Setting | Result |
| --- | --- |
| Capacity | Up to 2.3x/3.5x more requests than MuxServe++ in reported traces |
| Production A/B | 3.89x average throughput over several weeks with no SLO violations |
| Scale | Deployed across more than 10K GPUs |
| Per-request overhead | About 3 ms TTFT and 4 ms TPOT in one no-contention case |

## Limitation

- Balloon decisions depend on predicting working-set shifts.
- Host memory and activation bandwidth can become the next bottleneck.
- Global coordination introduces a larger failure/control surface.

---

*Reading date: 2026-08*
*Note status: Completed*
