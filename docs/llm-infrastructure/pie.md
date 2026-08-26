# Pie: A Programmable Serving System for Emerging LLM Applications

<div class="paper-meta" markdown>

**Authors**: In Gim, Zhiyao Ma, Seung-seob Lee, Lin Zhong  
**Institution**: Yale University  
**Conference**: SOSP '25  
**Year**: 2025  
**Paper Link**: [arXiv:2510.24051](https://arxiv.org/abs/2510.24051)  
**GitHub**: [pie-project/pie](https://github.com/pie-project/pie)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">programmable-serving</span>
<span class="paper-tag">agentic-workflows</span>
<span class="paper-tag">wasm</span>
</div>

## Background

Conventional engines own a monolithic prefill/decode loop and global KV policy. Agentic and reasoning applications need token-level branching, custom cache strategies, tool I/O, and application state, forcing invasive engine changes or slow client-side orchestration.

## Methodology

Pie decomposes inference into 42 service APIs. User programs called **inferlets** control embedding, forward, sampling, cache, messaging, HTTP, and workflow logic inside a WebAssembly sandbox.

| Layer | Responsibility |
| --- | --- |
| Application | Inferlet lifecycle and isolated application logic |
| Control | Cross-inferlet coordination and horizontal/vertical batching |
| Inference | GPU execution and command queues |

Trait-based APIs keep inferlets model-agnostic while allowing specialized forward methods to expose attention statistics. The server can batch compatible calls across different programs without taking control away from the application.

## Experiment

| Workload | Result |
| --- | --- |
| Standard generation | 3-12% latency overhead versus specialized serving engines |
| Agentic/reasoning workflows | 1.3x-3.4x better latency or throughput through application-specific optimization |

## Limitation

- Inferlets make application code responsible for policies previously centralized in the engine.
- WebAssembly isolation does not automatically guarantee semantic safety or fair resource use.
- Fine-grained APIs constrain future kernels unless the trait interface evolves with them.

---

*Reading date: 2026-08*
*Note status: Completed*
