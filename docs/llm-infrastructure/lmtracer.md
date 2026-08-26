# LMTracer: Fine-Grained and Real-Time Performance Profiling for Production LLM Systems

<div class="paper-meta" markdown>

**Authors**: Wei Liu, Yongchao He, Bohan Zhao, Hongyi Wang, Zhenhua Li, Junping Zhao  
**Institution**: SCITIX; Tsinghua University  
**Conference**: SOSP '26 (camera-ready not yet public)  
**Year**: 2026  
**Paper Link**: [Official project](https://llm-profiling.github.io/)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">profiling</span>
<span class="paper-tag">gpu-graphs</span>
<span class="paper-tag">production</span>
</div>

## Background

Coarse metrics cannot localize LLM performance regressions, while event tracing disrupts highly asynchronous GPU flows and is too expensive for always-on production use.

## Methodology

LMTracer compiles lightweight fence kernels as nodes inside the GPU execution graph. They read device clocks into shadow buffers and asynchronously ring a CPU doorbell only when data is ready, so user kernels continue without synchronous trace extraction. Plugins attach probes to vLLM, SGLang, Megatron, and TorchInductor at model, layer, function, or code-line granularity.

## Experiment

After more than four months in production, average overhead is 0.58% across training, serving, and fine-tuning. It proactively found 15 latent issues; fixes reduced latency by 21.2% and improved throughput by 24.5%.

## Limitation

- Graph-embedded probes require framework/runtime integration and GPU graph compatibility.
- Timing fences identify where time is spent, not necessarily why.
- This note uses the official project documentation; the final SOSP PDF is not public yet.

---

*Reading date: 2026-08*
*Note status: Completed from official project; paper pending*
