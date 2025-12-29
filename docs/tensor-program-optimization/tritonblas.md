# tritonBLAS: Triton-based Analytical Approach for GEMM Kernel Parameter Selection

<div class="paper-meta" markdown>

**Authors**: Ryan Swann, Muhammad Osama, Xiaohu Guo, Bryant Nelson, Lixun Zhang, Alex Brown, Yen Ong, Ali Yazdani, Sean Siddens, Ganesh Dasika, Alex Underwood
**Institution**: AMD
**Conference**: arXiv 2025
**Paper Link**: [arXiv:2512.04226](https://arxiv.org/abs/2512.04226)
**GitHub**: [ROCm/tritonBLAS](https://github.com/ROCm/tritonBLAS)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">GEMM Optimization</span>
<span class="paper-tag">Triton Compiler</span>
<span class="paper-tag">Analytical Model</span>
<span class="paper-tag">Zero Autotuning</span>
</div>

## Abstract

tritonBLAS presents a fast and deterministic analytical model that uses architectural parameters like cache hierarchy, and relative code and data placement to generate performant GPU GEMM kernels. It explicitly models the relationship between architectural topology, matrix shapes, and algorithmic blocking behavior to predict near-optimal configurations without runtime autotuning. The system achieves over 95% of autotuning performance while reducing autotuning time to zero.

**Key Contributions**:

- Analytical model for GEMM kernel parameter selection without autotuning
- Architecture-aware performance modeling using cache hierarchy and data locality
- 94.7% of exhaustive search efficiency with zero configuration overhead
- Over 95% performance compared to autotuning-based solutions
- Architecture portability with lightweight calibration (only need to update bandwidth/latency parameters)

## Core Ideas

tritonBLAS eliminates the need for expensive runtime autotuning by using an analytical approach to predict optimal GEMM kernel configurations. The model explicitly captures the interplay between GPU architecture (cache hierarchy, SM count), matrix shapes, tiling hierarchies, and data locality patterns. By modeling these relationships analytically, tritonBLAS can deterministically select near-optimal kernel parameters for any given matrix multiplication workload on AMD Instinct GPUs, achieving comparable performance to exhaustive autotuning search while drastically reducing configuration time.

## Design Goals

1. **Achieve Near-Optimal Performance**: Select GEMM parameters for near-optimal performance across wide range of matrix shapes
2. **No-Tuning Required**: General-purpose model applicable without exhaustive autotuning
3. **Lightweight and Fast**: Runtime model guidance without impacting performance
4. **Deterministic**: Same results for same inputs, ensuring reproducibility
5. **Architecture-Portable**: Agnostic to GPU micro-architecture, minimal adjustments needed

## Methodology

### 1. Hierarchical Tiling Structure

The model captures tiling structures arranged in a hierarchy from instruction-level to system-level:

| Memory Tiling Scope | Compute Scope | Physical Scope | Logical |
|---------------------|---------------|----------------|---------|
| Instruction | Matrix Core | Matrix Core | Instruction |
| Register | SIMD | SIMD | Wave |
| Shared Memory | Compute Unit | Compute Unit | Workgroup |
| L2 | Group of CU | XCD | None |
| LLC | All CUs | Device | None |

**Five Tiling Levels**:

1. **Instruction Level Tile**: Tile size consumed by matrix instruction (wmma, mfma) in each SIMD
2. **Warp/Wavefront (Register) Level Tile**: Set of instruction tiles processed by a SIMD over time
3. **Workgroup/Thread block (Shared Memory) Level Tile**: Tile stored in software managed memory between SIMDs
4. **Cache Tile**: Hardware managed memories with shared memory tiles in their scope
5. **Global Problem**: Total problem unrolled over space (CUs) and time

### 2. Quantifying Parallelism (Spatial Loop Unroll)

Three levels of parallelism are considered:

- **Matrix/Tensor Core-level**: Defined by GPU architects as latency of matrix instruction
- **Within Compute Unit**: Parallelized over SIMDs belonging to a CU
- **Across Compute Units (Occupancy)**: Utilization of all CUs in the system

**Key Insight**: Only the final timestep has "under-occupancy" (tail occupancy / wave quantization problem). Full occupancy can be assumed for all but the last timestep.

### 3. Quantifying Locality

Two types of data locality:

- **Software Managed**: Programmer-controlled memory (shared memory) with explicit data movement
- **Hardware Managed**: Caches with transparent data placement

**Reuse Analysis**: In workgroup tile, the $M \times K$ tile is reused $N$ times and the $N \times K$ tile is reused $M$ times.

### 4. Tradeoffs between Parallelism and Locality

The model identifies competing bottlenecks as rooflines:

- **Load/Store Issue Rate Bound**: Not enough load/store units occupied
    - *Solution*: Increase load/store unit occupancy
- **Software Managed Memory Bandwidth Bound**: Bound by loads from shared memory
    - *Solution*: Increase register tile size for better reuse
- **Cache Bandwidth Bound**: Limited by hardware cache bandwidth
    - *Solution*: Increase data reuse in software managed memory
- **Under-Occupied Compute Bound**: High utilization but low occupancy
    - *Solution*: Unroll K-loop or change tile size
- **Max Parallelism Compute Bound**: All CUs occupied with high matrix instruction utilization (optimal state)

## Key Formulas

### Algorithm 3: Compute Latency of Shared Memory Tile

$$N_{MI,M} = \lceil \frac{MT_M}{MI_M} \rceil, \quad N_{MI,N} = \lceil \frac{MT_N}{MI_N} \rceil, \quad N_{MI,K} = \lceil \frac{MT_K}{MI_K} \rceil$$

$$N_{MI} = N_{MI,M} \times N_{MI,N} \times N_{MI,K}$$

$$L_{MT} = L_{MI} \times N_{MI}$$

Where:
- $MI_M, MI_N, MI_K$: Dimensions of Matrix/Tensor Instruction
- $MT_M, MT_N, MT_K$: Dimensions of Shared Memory Tile
- $L_{MI}$: Latency of a matrix instruction
- $L_{MT}$: Compute latency per Shared Memory Tile

### Algorithm 4: Compute Active CUs in Last Timestep

$$n_{MT,M} = \lceil \frac{M}{MT_M} \rceil, \quad n_{MT,N} = \lceil \frac{N}{MT_N} \rceil$$

$$T_{out} = n_{MT,M} \times n_{MT,N}$$

$$\omega = \lceil \frac{T_{out}}{N_{CU}} \rceil$$

$$active\_cu = T_{out} \mod N_{CU}$$

Where $\omega$ is the number of waves and $active\_cu$ is the number of active CUs in the last wave.

### Algorithm 5: Estimate Cache Hit Rate

$$U = (m_t \cdot MT + n_t \cdot NT) \cdot KT \quad \text{(Uncached reads)}$$

$$R = (m_t \cdot n_t) \cdot (MT + NT) \cdot KT \quad \text{(Total reads)}$$

$$h = 1 - \frac{U}{R} \quad \text{(Hit rate)}$$

Where $m_t, n_t$ are cache-tile dimensions.

### Algorithm 7: Memory Latency of Loop Iteration

$$L_{CU\_lat} = \frac{Ld_{CU}}{R_{L1}}$$

$$T = Ld_{CU} \times C \quad \text{(Total loads across active CUs)}$$

$$L_1 = \frac{T}{R_1}, \quad T_2 = (1-H_1) \cdot T, \quad L_2 = \frac{T_2}{R_2}$$

$$T_M = (1-H_2) \cdot T_2, \quad L_{MEM} = \frac{T_M}{R_{MEM}} + L_{lat}$$

$$L_{mem} = \max(L_{CU\_lat}, L_1, L_2, L_{MEM})$$

Where $H_1, H_2$ are hit rates, $R_1, R_2, R_{MEM}$ are bandwidths.

### Algorithm 8: Output Tile Latency

$$L_{prologue} = L_{mem}$$

$$L_{epilogue} = \frac{a \cdot m_t \cdot n_t}{R_{mem}}$$

$$L_{loopiter} = \max(L_{compute}, L_{mem})$$

$$I = \lceil K/k_t \rceil - 1$$

$$L_{tile} = L_{prologue} + L_{epilogue} + (L_{loopiter} \times I)$$

### Algorithm 9: Total GEMM Latency

$$n_m = \lceil M/m_t \rceil, \quad n_n = \lceil N/n_t \rceil$$

$$N_{waves} = \lceil \frac{n_m \times n_n}{N_{CU}} \rceil$$

$$L_{total} = N_{waves} \times L_{tile}$$

## Performance Results

### Selection Efficiency
- **94.7%** selection efficiency relative to exhaustive search over 150,000 random problem sizes
- Most data points clustered near optimal, with more variation at lower arithmetic intensities (latency-bound problems)

### Selection Time Comparison

| Problem Size | Triton Autotuning (s) | tritonBLAS (s) |
|--------------|----------------------|----------------|
| 512 × 512 × 512 | 11.965 | 0.000070 |
| 1024 × 1024 × 1024 | 11.928 | 0.000057 |
| 4096 × 4096 × 4096 | 13.537 | 0.000055 |
| 16384 × 16384 × 16384 | 1383.594 | 0.000054 |

- **5-6 orders of magnitude faster** than autotuning (50-80 μs vs 10-50 seconds)
- tritonBLAS complexity: $O(T)$ vs Autotuning: $O(T \cdot M \cdot N \cdot K)$

### vs PyTorch
- On average **3% better** than PyTorch's matrix multiplication on MI300X
- On Llama3 matrix sizes: maximum 1.10× speedup, outperforming PyTorch in 10 cases
- Average 13.9% slower on Llama3 workloads (still competitive)

## Limitations

### Model Limitations

1. **Not a Comprehensive Latency Predictor**: Only captures latency "trends" for comparative evaluation, not absolute latency prediction

2. **Single-GPU Only**: Focuses on kernel implementations within a single GPU, not intended for distributed/multi-GPU environments

3. **GEMM Only**: Not intended for other GEMM-like algorithms such as attention mechanisms

4. **Simplified Cache Model**:
   - Does not model set associativity or replacement behavior
   - Only compares tile-level working-set to effective cache capacity
   - Limited influence on large, regular GEMM access patterns

### Implementation Limitations

5. **Triton Tile Constraints**: Triton does not support non-power-of-2 tile dimensions, limiting options for managing tile quantization (unlike vendor libraries like PyTorch which allow arbitrary increments)

6. **Selection Efficiency Gap**: ~5% average gap from optimal exhaustive search

7. **Performance Gap on Some Workloads**: Average 13.9% slower than PyTorch on Llama3 key matrices

### Autotuning Limitations Addressed

The paper identifies problems with autotuning that tritonBLAS solves:

- Dynamic tensor shapes varying run-to-run
- Real-time constraints requiring low latency (online inference, control systems)
- High cost/power consumption applications like LLM training

## Future Work

1. **Non-GEMM Workloads**: Explore extension to attention mechanisms and other GEMM-like algorithms

2. **Multi-GPU and Multi-Node**: Extend to distributed system environments

3. **Broader Algorithm Classes**: Foundation for extending analytical approach to other tensor operations

## Related Work

- **CUTLASS/cuBLAS**: Hand-engineered heuristics, closed-source, NVIDIA-specific
- **Ansor/AutoTVM**: Combine analytical estimates with learned correction terms, rely on data-driven fitting
- **DeLTA**: Locality-aware roofline predicting L1/L2 traffic, but analyzes predefined configurations rather than selecting them

**tritonBLAS Advantage**: Uses only calibrated hardware bandwidths and instruction latencies to directly rank and select tile shapes, enabling microsecond overhead and deterministic selection.

---

*Reading date: 2025*
*Note status: Completed*
