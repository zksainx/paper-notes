# HedraRAG: Coordinating LLM Generation and Database Retrieval in Heterogeneous RAG Serving

<div class="paper-meta" markdown>

**Authors**: Zhengding Hu, Vibha Murthy, Zaifeng Pan, Wanlu Li, Xiaoyi Fang, Yufei Ding, Yuke Wang  
**Institution**: UC San Diego; RegAilator; Rice University  
**Conference**: SOSP '25  
**Year**: 2025  
**Paper Link**: [arXiv:2507.09138](https://arxiv.org/abs/2507.09138)  
**GitHub**: [Leo9660/HedraRAG_AE](https://github.com/Leo9660/HedraRAG_AE)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">rag</span>
<span class="paper-tag">graph-runtime</span>
<span class="paper-tag">cpu-gpu-pipeline</span>
</div>

## Background

Advanced RAG alternates generation and vector retrieval over multi-step, IRG, HyDE, reranking, and compression workflows. Optimizing retrieval and generation independently leaves CPU/GPU pipeline bubbles and ignores cross-request skew.

## Methodology

RAGraph represents generation and retrieval as asymmetric graph nodes. Runtime transformations split nodes, reorder searches by semantic similarity, insert speculative edges, rewire dependencies, and cache hot index clusters on GPU. A scheduler forms wavefronts across active requests and maps them to CPU/GPU workers.

## Experiment

| Setting | Result |
| --- | --- |
| Online workflows | More than 1.5x and up to 5x throughput improvement |
| Offline | 3.5x over LangChain and 1.3x over FlashRAG |
| Speculative retrieval/generation | 1.06x-1.62x latency speedup |
| GPU index caching | 1.12x-1.49x speedup depending on hotspot reuse |

## Limitation

- Benefits shrink when the LLM dominates and the database is small.
- Speculation can roll back; index caching depends on access skew.
- Graph cost estimates must track changing model and retrieval latency.

---

*Reading date: 2026-08*
*Note status: Completed*
