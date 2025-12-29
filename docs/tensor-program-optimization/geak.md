# GEAK: Introducing Triton Kernel AI Agent & Evaluation Benchmarks

<div class="paper-meta" markdown>

**Authors**: Jianghui Wang, Vinay Joshi, Saptarshi Majumder, Xu Chao, Bin Ding, Ziqiong Liu, Pratik Prabhanjan Brahma, Dong Li, Zicheng Liu, Emad Barsoum  
**Institution**: AMD  
**Conference**: arXiv 2025  
**Paper Link**: [arXiv:2507.23194](https://arxiv.org/abs/2507.23194)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Triton</span>
<span class="paper-tag">LLM Agent</span>
<span class="paper-tag">GPU Kernel Generation</span>
<span class="paper-tag">AMD GPU</span>
</div>

## Abstract

GEAK (Generating Efficient AI-centric GPU Kernels) is AMD's agentic framework for automatic Triton kernel generation targeting AMD Instinct GPUs. It leverages inference-time compute scaling with frontier LLMs to produce Triton-based GPU kernels using a reasoning loop adapted from Reflexion-style feedback mechanisms.

**Key Contributions**:
- GEAK framework: modular, agent-based system for Triton kernel generation from minimal task descriptions
- Two benchmark suites: TritonBench-revised (184 kernels) and ROCm Triton Benchmark (30 real-world kernels)
- Achieves up to 63% execution accuracy and 2.59x speedup over reference implementations
- Open-sourced agent implementation and evaluation framework

## Core Ideas

### GEAK Pipeline

Four core modules:
1. **Generator**: Produces code based on user query and contextual information
2. **Evaluator**: Cascaded design - functionality test first, then performance assessment (latency, memory efficiency)
3. **Reflector**: Analyzes failed code and error traces to identify issues
4. **Optimizer**: Takes functionally correct code and formulates performance enhancement strategies

### Key Techniques

**1-shot Prompting**: Uses most similar Triton code from datasets (code similarity more effective than instruction similarity)

**Knowledge Injection**: Enhances prompt with domain-specific knowledge on efficient Triton kernels including hardware specifications

**Reflexion**: When generated kernel fails functionality test, error trace is provided to reflector for iterative refinement

**LLM as Optimizer**: Identifies optimization directions based on previous code generations sorted by performance

**Debugging Trap Prevention**: Limits debugging attempts per code snippet (`max_perf_debug_num`); if exceeded, discard current approach and generate fresh code

**Parallel Scaling**: Multiple GEAK instances run in parallel with temperature=1 for diversity; combined with sequential scaling yields additional improvements

### Benchmarks

**TritonBench-revised**: 184 kernels adapted from TritonBench-G with stricter testing:
- Fixed 37 kernels with AMD GPU errors
- Added tolerance-based tensor comparison instead of STDOUT string comparison
- Ensured consistent seed for random tensor generation

**ROCm Triton Benchmark**: 30 real-world kernels from open-source AMD repositories:
- ROCm/triton, ROCm/aiter, ROCm/aotriton, ROCm/vllm, ROCm/pytorch, ROCm/xformers, etc.

### Evaluation Metrics

- **Call Accuracy**: Fraction that compile and run without errors
- **Execution Accuracy**: Percentage satisfying all unit tests
- **Speedup**: Ratio of reference kernel latency to generated kernel latency

## Experimental Results

**Baseline (Direct LLM Prompting)**:
| Model | Call Acc (0/1-shot) | Exec Acc (0/1-shot) | Speedup |
|-------|---------------------|---------------------|---------|
| GPT4.1 | 14.67% / 19.02% | 8.70% / 14.13% | 0.52 / 0.53 |
| Gemini 2.5 Pro | 20.65% / 21.74% | 14.13% / 16.85% | 1.33 / 0.96 |

**GEAK Results on TritonBench-revised (MI300)**:
| Difficulty | Total | Exec Accuracy | Avg. Speedup |
|------------|-------|---------------|--------------|
| Level 1 | 3 | 100.0% | 1.16x |
| Level 2 | 27 | 81.48% | 1.69x |
| Level 3 | 65 | 63.08% | 3.02x |
| Level 4 | 84 | 40.48% | 2.86x |
| **Overall** | **184** | **54.89%** | **2.59x** |

**GEAK on ROCm Benchmark**: 63.33% execution accuracy, 0.92x speedup

**Sequential Scaling Effect** (TritonBench on MI250):
- iter0: 13.04% exec accuracy → iter19: 44.02% exec accuracy
- 3x+ improvement through iterative refinement

**Module Ablation**:
| Knowledge | 1-shot | Optimizer | Call Acc | Exec Acc | Speedup |
|-----------|--------|-----------|----------|----------|---------|
| - | - | - | 14.67% | 8.70% | 0.52 |
| ✓ | - | - | 52.72% | 20.11% | 0.86 |
| ✓ | ✓ | - | 54.35% | 27.17% | 0.99 |
| ✓ | ✓ | ✓ | 56.52% | 40.76% | 1.45 |

## Case Study: Flip Kernel (2.26x speedup)

**Expert-written code limitations**:
- Double memory access pattern (load → flip in registers → store)
- High register pressure holding entire block
- Limited flexibility with `tl.flip()` behavior

**GEAK optimization**:
- Single-pass operation: read from flipped positions, write directly to destination
- Lower register usage, better cache efficiency
- Explicit masking for boundary conditions
- Coalesced memory access pattern

---

*Reading date: 2025-12*
*Note status: Completed*
