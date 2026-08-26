# WaferLLM: Large Language Model Inference at Wafer Scale

<div class="paper-meta" markdown>

**Authors**: Congjie He, Yeqi Huang, Pei Mu, Ziming Miao, Jilong Xue, Lingxiao Ma, Fan Yang, Luo Mai  
**Institution**: University of Edinburgh; Microsoft Research  
**Conference**: OSDI '25  
**Year**: 2025  
**Paper Link**: [USENIX paper](https://www.usenix.org/conference/osdi25/presentation/he)  
**GitHub**: [MeshInfra/WaferLLM](https://github.com/MeshInfra/WaferLLM)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">wafer-scale</span>
<span class="paper-tag">llm-inference</span>
<span class="paper-tag">memory-bandwidth</span>
</div>

## Background

Wafer-scale accelerators expose hundreds of thousands of cores, tens of GB of distributed on-chip memory, and tens of PB/s of bandwidth. Their mesh-local-memory architecture is fundamentally different from the shared-memory GPU model assumed by systems such as vLLM and SGLang. LLM decode is especially suitable for this hardware because each token generation repeatedly performs GEMV and is dominated by weight movement.

**Key Takeaways**

- The central contribution is a hardware/software contract, not merely a faster kernel: PLMR models massive parallelism, non-uniform latency, limited local memory, and constrained routing.
- WaferLLM maps prefill and decode differently, using fine-grained parallelism for GEMM and locality-aware placement for GEMV.
- MeshGEMM and MeshGEMV are specialized implementations that make the placement policy executable on Cerebras WSE-2.

## Methodology

### System Overview

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| PLMR model | WSE topology and memory parameters | Placement constraints | Rejects GPU-style assumptions |
| Wafer-scale parallelism | Transformer layers and tensors | Mesh partitions | Places computation/data under local-memory and routing limits |
| MeshGEMM | Prefill matrices | Distributed output | Exploits massive core parallelism |
| MeshGEMV | Decode weights and activations | Token logits | Minimizes long-distance traffic and reuses local data |

The implementation uses about 7,000 lines of Cerebras CSL plus 2,000 lines of Python for checkpoint loading, launch, and policy selection. The design keeps the whole model on one wafer where possible, reducing the off-chip communication that dominates multi-GPU inference.

## Experiment

| Setting | Result |
| --- | --- |
| Accelerator utilization | Up to 200x higher than prior wafer-scale methods |
| GEMV | 606x faster and 16x more energy-efficient than an NVIDIA A100 in the reported comparison |
| End-to-end inference | 10-20x faster than A100 GPU clusters running SGLang/vLLM |
| Energy | About 2-2.5x better than A100 for the reported WSE-2 decode cases |

The evaluation covers Llama2/Llama3 models, prefill and decode separately, and full inference on Cerebras WSE-2. The paper also reports that a single A100 can remain more energy efficient for some small cases, because WSE-2's large fixed device power is amortized only at sufficient utilization.

## Limitation

- Results depend on Cerebras-specific topology, compiler, and device availability.
- The PLMR abstraction is useful for this architecture but does not directly predict GPU-cluster behavior.
- Model placement can lose efficiency when layers or communication patterns do not fit the wafer mesh cleanly.

---

*Reading date: 2026-08*
*Note status: Completed*
