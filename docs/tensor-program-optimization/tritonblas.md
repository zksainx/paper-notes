# tritonBLAS: Triton-based Analytical Approach for GEMM Kernel Parameter Selection

<div class="paper-meta" markdown>

**Authors**: Ryan Swann, Muhammad Osama, Xiaohu Guo  
**Institution**: AMD  
**Conference**: arXiv 2025  
**Paper Link**: [arXiv:2512.04226](https://arxiv.org/abs/2512.04226)  
**GitHub**: [ROCm/tritonBLAS](https://github.com/ROCm/tritonBLAS)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">GEMM Optimization</span>
<span class="paper-tag">Triton Compiler</span>
<span class="paper-tag">Analytical Model</span>
</div>

## Abstract

tritonBLAS presents a fast and deterministic analytical model that uses architectural parameters like cache hierarchy, and relative code and data placement to generate performant GPU GEMM kernels. It explicitly models the relationship between architectural topology, matrix shapes, and algorithmic blocking behavior to predict near-optimal configurations without runtime autotuning. The system achieves over 95% of autotuning performance while reducing autotuning time to zero.

**Key Contributions**:
- Analytical model for GEMM kernel parameter selection without autotuning
- Architecture-aware performance modeling using cache hierarchy and data locality
- 94.7% of exhaustive search efficiency with zero configuration overhead
- Over 95% performance compared to autotuning-based solutions

## Core Ideas

tritonBLAS eliminates the need for expensive runtime autotuning by using an analytical approach to predict optimal GEMM kernel configurations. The model explicitly captures the interplay between GPU architecture (cache hierarchy, SM count), matrix shapes, tiling hierarchies, and data locality patterns. By modeling these relationships analytically, tritonBLAS can deterministically select near-optimal kernel parameters for any given matrix multiplication workload on AMD Instinct GPUs, achieving comparable performance to exhaustive autotuning search while drastically reducing configuration time.

---

*Reading date: 2025*
*Note status: Completed*
