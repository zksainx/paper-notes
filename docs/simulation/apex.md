# APEX: An Extensible and Dynamism-Aware Simulator for Automated Parallel Execution in LLM Serving

<div class="paper-meta" markdown>

**Authors**: Yi-Chien Lin, Woosuk Kwon, Ronald Pineda  
**Institution**: USC, UC Berkeley, UCLA  
**Conference**: arXiv 2024  
**Paper Link**: [arXiv:2411.17651](https://arxiv.org/abs/2411.17651)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">LLM Serving</span>
<span class="paper-tag">Parallel Execution</span>
<span class="paper-tag">Performance Simulation</span>
</div>

## Abstract

APEX is an LLM serving system simulator that efficiently identifies optimal parallel execution plans by considering key factors such as memory usage and batching behavior. The simulator performs dynamism-aware simulation to model iteration-level batching and leverages LLMs' repetitive structure to reduce design space exploration.

**Key Contributions**:
- Dynamism-aware simulation for iteration-level batching in LLM serving
- Efficient design space exploration leveraging LLM repetitive structure
- Scales to trillion-scale models with optimal parallel execution planning

## Core Ideas

APEX addresses the challenge of selecting optimal parallel execution plans by balancing computation, memory, and communication overhead. It efficiently handles varying parallelism techniques (data, pipeline, tensor) and workload characteristics to identify the best execution strategy.

---

*Reading date: 2025*
*Note status: Completed*
