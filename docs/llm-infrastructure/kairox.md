# Kairox: Adaptive GPU-CPU Hybrid LLM Inference via Online Neuron Balancing

<div class="paper-meta" markdown>

**Authors**: Yapeng Jiang, Minghao Gan, Zicong Hong, Wuhui Chen, Junyuan Liang, Yue Yu, Meng Guo, Zibin Zheng  
**Institution**: Sun Yat-Sen University; EPFL; Peng Cheng Laboratory; Qilu University of Technology  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/jiang-yapeng)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">activation-sparsity</span>
<span class="paper-tag">neuron-placement</span>
<span class="paper-tag">hybrid-inference</span>
</div>

## Background

Static hot/cold FFN neuron placement cannot follow input-dependent activation, leaving the CPU bottlenecked or wasting GPU space.

## Methodology

Kairox predicts next-layer activation in a live pipeline, migrates neurons online, uses Temporal Activation Momentum to retain consistently useful neurons, and adjusts migration intensity based on current CPU/GPU/PCIe bottlenecks.

## Experiment

It improves throughput by up to 7.57x, 3.70x, 6.35x, and 3.76x over llama.cpp, PowerInfer, Neuralink, and Q-Infer; geomean gain is 3.15x/3.93x over llama.cpp on two PCs and about 2.1x over sparse baselines.

## Limitation

- Prediction misses and transient activation cause wasted transfers.
- Online migration competes for PCIe bandwidth.
- Requires activation-sparse model execution support.

---

*Reading date: 2026-08*
*Note status: Completed*
