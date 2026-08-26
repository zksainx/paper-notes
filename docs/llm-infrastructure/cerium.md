# Cerium: A Multi-GPU Framework for Terabyte-Scale Encrypted Inference

<div class="paper-meta" markdown>

**Authors**: Siddharth Jayashankar, Joshua Kim, Michael B. Sullivan, Wenting Zheng, Dimitrios Skarlatos  
**Institution**: Carnegie Mellon University; UT Austin; NVIDIA  
**Conference**: SOSP '26  
**Year**: 2026  
**Paper Link**: [arXiv:2512.11269](https://arxiv.org/abs/2512.11269)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">homomorphic-encryption</span>
<span class="paper-tag">multi-gpu</span>
<span class="paper-tag">compiler-runtime</span>
</div>

## Background

FHE expands BERT weights to about 1.5 TB and Llama3-8B to 112 TB in encoded form. Thousands of polynomial kernels, intermediate ciphertexts, launch overhead, and host/GPU orchestration make encrypted large-model inference infeasible with existing GPU libraries.

## Methodology

Cerium combines an FHE DSL/IR, horizontal/vertical kernel fusion, NTT/base-conversion kernels, lifetime-based memory reuse, sparse symmetric plaintext encoding, multi-GPU limb/program parallelism, compute/communication overlap, CUDA graphs, and guided UVM prefetch.

## Experiment

Plaintext compression shrinks BERT by 96x and Llama3-8B by 119x (112 TB to 982 GB). Cerium runs encrypted BERT in 8.8 s and Llama3-8B in 134 s, bootstraps in 7.5 ms, and reaches 1-4.4x the performance of evaluated FHE ASICs. Multi-GPU Cerium is 2.24x Cheddar; guided memory management gives 12.1x on Llama3-8B.

## Limitation

- Even optimized encrypted Llama inference remains orders slower than plaintext.
- Compilation and CUDA-graph creation are substantial for new circuits.
- Security/parameter selection and model accuracy are separate from systems optimization.

---

*Reading date: 2026-08*
*Note status: Completed*
