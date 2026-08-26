# StriaTrace: Efficient Tracing and Diagnosis for Online LLM Inference

<div class="paper-meta" markdown>

**Authors**: Haonan Wu, Yanqing Chen, Kun Qian, Xue Li, Jingbo Xu, Erci Xu, Ennan Zhai, Wenyuan Yu, Guangtao Xue, Jingren Zhou  
**Institution**: Shanghai Jiao Tong University; Alibaba Group  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/wu-haonan)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">observability</span>
<span class="paper-tag">llm-inference</span>
<span class="paper-tag">diagnosis</span>
</div>

## Background

Sporadic TTFT/TPOT anomalies violate inference SLOs, but continuous Nsight-like tracing costs 10-20%, and training-oriented diagnosis misses streaming request critical paths.

## Methodology

StriaTrace continuously records synchronization points and critical paths, escalating to detailed tracing only around anomalies. A dynamic regression roofline attributes performance ceilings; correlation analysis ranks likely causes.

## Experiment

It reduces tracing overhead by 97.8% relative to alternatives and has diagnosed hundreds of anomalies spanning 19 root-cause classes across development, testing, and production release cycles.

## Limitation

- Regression/correlation identifies likely causes rather than proving causality.
- Selective detail can miss precursors outside the trigger window.
- Instrumentation must evolve with serving architecture changes.

---

*Reading date: 2026-08*
*Note status: Completed*
