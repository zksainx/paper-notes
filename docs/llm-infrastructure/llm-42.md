# LLM-42: Enabling Determinism in LLM Inference with Verified Speculation

<div class="paper-meta" markdown>

**Authors**: Raja Gond, Aditya K. Kamath, Ramachandran Ramjee, Ashish Panwar  
**Institution**: Microsoft Research; University of Washington  
**Conference**: SOSP '26  
**Year**: 2026  
**Paper Link**: [arXiv:2601.17768](https://arxiv.org/abs/2601.17768)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">deterministic-inference</span>
<span class="paper-tag">verify-rollback</span>
<span class="paper-tag">dynamic-batching</span>
</div>

## Background

Floating-point reduction order changes with batch shape, so identical prompts can diverge. Batch-invariant kernels impose determinism globally and lose optimized split-K/fusion: one deterministic request in an 11-request batch cuts SGLang throughput from 931 to about 415 tokens/s.

## Methodology

LLM-42 uses a fast non-deterministic decode path plus deterministic decode-verify-rollback. A fixed-size verifier replays candidate windows under shape-consistent reductions, commits the matching prefix, repairs KV entries from the verifier, and rolls back the rest. Determinism is selected per request; non-deterministic traffic retains normal kernels.

## Experiment

With one deterministic request among eleven, it reaches 911 tokens/s: 2.2x deterministic SGLang and within 3% of the non-deterministic path. At low deterministic ratios, throughput stays within roughly 1-8% of the best case; window size trades verification cost (0.75 to 0.05 ms/token) against recomputation.

## Limitation

- Every operator in the verifier needs a consistent implementation/configuration.
- Long verification windows can cause 46% recomputation under mismatch.
- It guarantees repeatable execution, not semantic correctness of the answer.

---

*Reading date: 2026-08*
*Note status: Completed*
