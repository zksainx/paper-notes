# METIS: Fast Quality-Aware RAG Systems with Configuration Adaptation

<div class="paper-meta" markdown>

**Authors**: Siddhant Ray, Rui Pan, Zhuohan Gu, Kuntai Du, Shaoting Feng, Ganesh Ananthanarayanan, Ravi Netravali, Junchen Jiang  
**Institution**: University of Chicago; Princeton University; TensorMesh; Microsoft  
**Conference**: SOSP '25  
**Year**: 2025  
**Paper Link**: [arXiv:2412.10543](https://arxiv.org/abs/2412.10543)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">rag</span>
<span class="paper-tag">configuration-adaptation</span>
<span class="paper-tag">quality-latency</span>
</div>

## Background

RAG knobs such as retrieval count, synthesis method, and intermediate length have query-specific quality/latency effects. Retrieving more documents can reduce quality by up to 20% while tripling delay, so a single “high quality” static configuration is neither fast nor reliably better.

## Methodology

METIS uses an LLM profiler to classify query complexity, prunes the configuration space by 50-100x, maps profiles to configurations, and jointly schedules queries under memory/load constraints. The profiler contributes only 3-6% average delay in the reported datasets.

## Experiment

| Metric | Result |
| --- | --- |
| Generation latency | 1.64x-2.54x reduction without quality loss across four datasets |
| Low-load no-batching cases | 1.48x-1.56x lower average delay |
| Scheduler contribution | Approximately 1.45x-1.75x delay reduction beyond adaptation alone |

## Limitation

- Quality estimators and mapping rules require calibration as data/models change.
- The profiler itself is an LLM dependency with cost and failure modes.
- The supported knob space is curated rather than automatically discovered.

---

*Reading date: 2026-08*
*Note status: Completed*
