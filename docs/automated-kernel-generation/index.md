# Automated Kernel Generation

Papers on LLM-driven generation, profiling-guided refinement, and automated optimization of GPU kernels and low-level code.

## Research Areas

- LLM-guided GPU kernel synthesis
- Profiling-aware iterative optimization
- Agentic compiler and kernel refinement loops
- Automated autotuning and search with execution feedback
- Benchmarking and evaluation for generated kernels

## Papers

### NPU Kernel Generation

- [AscendKernelGen](ascendkernelgen.md) - Domain-adaptive framework for Ascend NPU kernel generation with reasoning data, post-training, and hardware-grounded evaluation

### Evolutionary Population Design

- [EvoEngineer](evoengineer.md) - Systematic LLM-based code evolution framework that separates traverse strategy from population management for CUDA kernel optimization

### Heterogeneous Accelerator Systems

- [KernelEvolve](kernelevolve.md) - Agentic graph-search framework for generating and optimizing kernels across NVIDIA, AMD, and Meta MTIA accelerators

### Skill and Memory Retrieval

- [KernelSkill](kernelskill.md) - Multi-agent kernel optimizer with long-term expert skill memory and short-term trajectory memory

### World-Model Search

- [K-Search](k-search.md) - Co-evolving world-model search framework that separates high-level optimization planning from low-level kernel instantiation

### Evolutionary Agentic Search

- [AVO](avo.md) - Agentic variation operator that replaces fixed mutation and crossover with an autonomous coding agent for long-running kernel evolution

### Persistent Memory and Continual Learning

- [KernelBlaster](kernelblaster.md) - Memory-augmented in-context RL workflow that accumulates CUDA optimization knowledge across tasks and hardware generations

### Bandit and Search Policies

- [KernelBand](kernelband.md) - Hardware-aware contextual bandit framework for steering Triton kernel optimization across diverse GPUs and LLM backends

### Agentic Triton Generation on AMD

- [GEAK](geak.md) - Modular Triton kernel agent for AMD GPUs with revised benchmarks, Reflexion-style debugging, and optimization modules

### Search and Reinforcement Frameworks

- [CUDA-LLM](cuda-llm.md) - Feature Search and Reinforcement loop that iteratively repairs and accelerates CUDA kernels using compilation, correctness, and runtime feedback

### Natural-Language Transformation Systems

- [PEAK](peak.md) - Natural-language transformation framework for iterative GPU kernel performance engineering across CUDA, HIP, and HLSL

### Profiling-Guided Optimization

- [TritonForge](tritonforge.md) - Iterative Triton kernel optimization with LLM agents, Nsight feedback, and automated remediation

---

*Continuously updated...*
