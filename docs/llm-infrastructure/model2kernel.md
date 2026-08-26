# Model2Kernel: Model-Aware Symbolic Execution for Safe CUDA Kernels

<div class="paper-meta" markdown>

**Authors**: Mengting He, Shihao Xia, Haomin Jia, Wenfei Wu, Linhai Song  
**Institution**: Penn State; ICT CAS; Peking University  
**Conference**: SOSP '26  
**Year**: 2026  
**Paper Link**: [arXiv:2603.24595](https://arxiv.org/abs/2603.24595)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">cuda-verification</span>
<span class="paper-tag">symbolic-execution</span>
<span class="paper-tag">memory-safety</span>
</div>

## Background

LLM kernels combine model-fixed layout constraints with user-controlled sequence/batch shapes. Generic symbolic execution invents infeasible inputs; dynamic sanitizers require triggering inputs and add runtime overhead.

## Methodology

HFProbe profiles real model/framework invocation and classifies arguments; configuration mutation explores legal model shapes. cuKLEE symbolically represents tensors, block/thread IDs, dynamic memory, and CUDA synchronization to find integer overflow, OOB, races, and null accesses under model-derived constraints.

## Experiment

Across vLLM, Hugging Face, and research kernels, Model2Kernel finds 353 previously unknown bugs with nine false positives. Removing model profiling creates 5,527 false positives; without configuration mutation only 21/50 kernels execute and 32 bugs are missed.

## Limitation

- Each kernel has a one-hour analysis cap and some known bugs time out.
- Unsupported CUDA semantics and inaccurate model weights/constraints cause misses or false positives.
- Memory safety does not establish numerical correctness.

---

*Reading date: 2026-08*
*Note status: Completed*
