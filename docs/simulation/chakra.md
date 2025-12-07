# Chakra: Execution Traces for Benchmarking and Network Performance Optimization

<div class="paper-meta" markdown>

**Authors**: Meta AI Research  
**Institution**: Meta (Facebook)  
**Conference**: Engineering at Meta 2023  
**Paper Link**: [Meta Engineering Blog](https://engineering.fb.com/2023/09/07/networking-traffic/chakra-execution-traces-benchmarking-network-performance-optimization/)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Execution Traces</span>
<span class="paper-tag">Benchmarking</span>
<span class="paper-tag">Performance Analysis</span>
</div>

## Abstract

Chakra introduces an open graph-based representation of AI/ML workload execution, laying the foundation for benchmarking and network performance optimization. It provides a standardized schema for performance modeling through execution traces.

**Key Contributions**:
- Graph-based representation of AI/ML workload execution
- Standardized Chakra execution trace format
- Captures compute, communication, and memory operations with timing and dependencies
- Integration with MLCommons for industry-wide adoption

## Core Ideas

Chakra execution traces capture arbitrary distributed ML workloads using a directed acyclic graph (DAG) representation of compute, communication, and remote memory nodes. In collaboration with MLCommons, Meta has open-sourced tools to enable collection, analysis, generation, and adoption of Chakra traces by simulators, emulators, and replay tools.

---

*Reading date: 2025*
*Note status: Completed*
