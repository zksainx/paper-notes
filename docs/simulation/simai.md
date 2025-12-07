# SimAI: Unifying Architecture Design and Performance Tuning for Large-Scale Large Language Model Training with Scalability and Precision

<div class="paper-meta" markdown>

**Authors**: Xizheng Wang, Qingxu Li, Yichi Xu  
**Institution**: Alibaba Cloud, Tsinghua University  
**Conference**: NSDI 2025  
**Paper Link**: [USENIX](https://www.usenix.org/conference/nsdi25/presentation/wang-xizheng-simai) | [GitHub](https://github.com/aliyun/SimAI)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">LLM Training</span>
<span class="paper-tag">Architecture Design</span>
<span class="paper-tag">Performance Tuning</span>
</div>

## Abstract

SimAI is the industry's first full-stack, high-precision simulator for AI large-scale training. It aims at precisely and efficiently simulating the LLM training procedure at scale, achieving an average of 98.1% alignment to real-world results under various test scenarios.

**Key Contributions**:
- High-fidelity integration of training frameworks, kernel computation, and collective communication
- Multi-thread acceleration with lock-free global context-sharing
- Three operation modes: Analytical, Simulation, and Physical
- Comprehensive modeling of the entire LLM training process

## Core Ideas

SimAI supports three major operation modes: (1) SimAI-Analytical for fast simulation using bus bandwidth abstraction, (2) SimAI-Simulation for full-stack simulation with fine-grained network communication modeling using NS3, and (3) SimAI-Physical for physical traffic generation in CPU RDMA cluster environments. The simulator provides detailed modeling of framework, collective communication, and network layers.

---

*Reading date: 2025*
*Note status: Completed*
