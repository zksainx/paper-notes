# ASTRA-sim: Enabling SW/HW Co-Design Exploration for Distributed DL Training Platforms

<div class="paper-meta" markdown>

**Authors**: Collaboration between Intel, Meta, and Georgia Tech  
**Institution**: Intel, Meta, Georgia Tech  
**Conference**: IEEE/Multiple Conferences  
**Paper Link**: [ASTRA-sim Website](https://astra-sim.github.io/)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">Distributed Training</span>
<span class="paper-tag">Network Simulation</span>
<span class="paper-tag">Hardware-Software Co-design</span>
</div>

## Abstract

ASTRA-sim is a distributed machine learning system simulator that models the end-to-end software and hardware stack of modern AI systems, including workload scheduling, collective communication algorithms, and hardware architectures (compute/memory/network).

**Key Contributions**:
- End-to-end cycle-accurate distributed training simulator
- Support for multiple compute models (roofline, SCALE-sim) and network models (analytical, Garnet, NS3)
- Enables systematic study of bottlenecks for scaling training of large models

## Core Ideas

ASTRA-sim enables design-space exploration for running large DNN models over future training platforms. It supports simulation from simple analytical to detailed cycle-accurate modeling of large-scale training platforms, particularly relevant for models like GPT-3 or DLRM that need to scale across thousands of accelerator nodes.

---

*Reading date: 2025*
*Note status: Completed*
