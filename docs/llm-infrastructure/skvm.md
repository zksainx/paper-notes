# SkVM: A Language VM for Skills across Heterogeneous LLMs and Harnesses

<div class="paper-meta" markdown>

**Authors**: Le Chen, Erhu Feng, Yubin Xia, Haibo Chen  
**Institution**: Shanghai Jiao Tong University  
**Conference**: SOSP '26  
**Year**: 2026  
**Paper Link**: [arXiv:2604.03088](https://arxiv.org/abs/2604.03088)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">agent-skills</span>
<span class="paper-tag">compilation</span>
<span class="paper-tag">portability</span>
</div>

## Background

Skills are raw natural-language/script context whose success varies by model and agent harness. Analysis of 118K skills shows that adding a skill can even degrade task success because target capabilities, environment, and concurrency semantics differ.

## Methodology

SkVM treats a skill as code and a model-harness pair as a heterogeneous processor. It profiles primitive capabilities, compiles requirements to the target, binds tools/environment, extracts concurrency, JIT-solidifies stable code paths, and recompiles adaptively after failures/feedback.

## Experiment

Across eight LLMs, three harnesses, SkillsBench, and representative tasks, SkVM improves completion while reducing tokens up to 40%. Concurrency gives up to 3.2x speedup; code solidification reduces latency by 19-50x.

## Limitation

- Capability profiling and compilation must be repeated as models/harnesses change.
- Solidified code may freeze incorrect behavior or lose adaptability.
- Natural-language skill semantics remain hard to specify precisely.

---

*Reading date: 2026-08*
*Note status: Completed*
