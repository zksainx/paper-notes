# UniCoX: A Unified Cost Model for Tensorized Program Tuning Across Ubiquitous Accelerators

<div class="paper-meta" markdown>

**Authors**: Zihan Wang, Lei Gong, Wenqi Lou, Teng Wang, Qianyu Cheng, Xianglan Chen, Chao Wang, Xuehai Zhou  
**Institution**: University of Science and Technology of China, Suzhou Institute for Advanced Research  
**Conference**: IEEE Transactions on Computers, Vol. 75, No. 1, January 2026  
**Paper Link**: [IEEE TC](https://doi.org/10.1109/TC.2025.3626148)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Cost Model</span>
<span class="paper-tag">Tensorized Programs</span>
<span class="paper-tag">Transfer Learning</span>
<span class="paper-tag">Lifelong Learning</span>
</div>

## Abstract

UniCoX is a unified cost model specifically designed for tensorized program tuning across diverse hardware accelerators. It addresses the lack of cost models for tensorized programs (programs that leverage hardware intrinsics like Tensor Cores, AVX512, NEON) by proposing unified feature representation and transfer prediction strategies.

**Key Contributions**:
- Unified cross-accelerator feature representation for tensorized programs using attention-based AST feature mining
- Transfer prediction strategy combining lifelong learning (cross-semantic) and transfer learning (cross-performance)
- TensorizeSetX dataset covering 18 intrinsics across 6 hardware platforms
- Achieves 11.3x search time speedup and 1.9x inference speed improvement in TVM MetaSchedule

## Core Ideas

### Problem: Tensorized vs Normal Programs

| Aspect | Normal Tensor Program | Tensorized Program |
|--------|----------------------|-------------------|
| Execution | Arithmetic Operations | Hardware Intrinsics |
| Data Type | Float32 | Mixed Precision |
| Rate of Change | Gradual | Rapid Growth |
| Diversity | Platform only | Intrinsic + Platform |
| Behavior | Freedom | Relatively Fixed |

### Three Design Requirements

**Feature Representation**:
1. **Req 1**: Extract key software schedule features affecting performance
2. **Req 2**: Design intrinsic representations aligned with key schedules
3. **Req 3**: Unified representation across hardware platforms

**Transfer Prediction**:
4. **Req 4**: Cross-semantic transfer (new intrinsic semantics)
5. **Req 5**: Cross-performance transfer (microarchitecture upgrades)
6. **Req 6**: Efficient data sampling strategies

### Unified Feature Representation

Uses a behavior template abstracting tensorized programs into three stages:
- **Load**: Read data through multi-level caches
- **Compute**: Execute tensorized operations via intrinsics
- **Store**: Write results back to memory

Each behavior represented by three vectors:
- `loopnest`: loop structures (loopid, extent, schedule)
- `affine`: index expressions
- `buffer/intrinsic`: data access or intrinsic configuration

### Key Schedule Mining via Attention

Applied TLP (Token-Level Performance) to tensorized programs (TDTLP) and analyzed attention matrices. Key schedules identified:
- `Split`, `SamplePerfectTile`: Loop tiling
- `SampleCategorical`: Vectorization, unroll, parallel binding
- `CacheWrite`, `ComputeInline`: Behavior modification

### Transfer Prediction Strategy

**Cross-Semantic (Lifelong Learning)**:
- Uses EWC (Elastic Weight Consolidation) to prevent catastrophic forgetting
- Loss function: $L'(\theta) = L(\theta) + \lambda \sum_i b_i(\theta_i - \theta_i^b)^2$
- Data sampling: Prioritize subgraphs with more intrinsic calls

**Cross-Performance (Transfer Learning)**:
- Pre-train on low-performance implementation, fine-tune on high-performance
- Data sampling: Prioritize subgraphs with significant ranking shifts (Kendall tau)

### Model Architecture

- Behavior-aligned encoding with 2n+1 fully connected layers
- Attention layer for learning behavior relationships
- Lambda rank loss for training

## Experimental Results

**Platforms**: NVIDIA A100/T4 GPU, Intel Xeon 8488C/8369B, ARM Yitian710, FPGA PYNQ

**Key Results**:
- State-of-the-art prediction accuracy (Top-1: 0.93, Top-5: 0.99 on some tasks)
- Cross-semantic transfer maintains accuracy on old semantics while learning new ones
- Cross-performance transfer achieves comparable accuracy with 40-50% training data
- Search-based: 11.3x faster tuning time, 1.9x inference speedup vs online model

## Limitations & Future Work

- Current approach focuses on performance estimation; future work aims for direct prediction of optimal intrinsic + schedule combinations
- End-to-end decision making without exhaustive search

---

*Reading date: 2025-12*
*Note status: Completed*
