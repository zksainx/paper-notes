# ADAngel: Accelerating Arbitrary-Precision Quantized LLMs with Adaptive Computing Mapping

<div class="paper-meta" markdown>

**Authors**: Yao Liu, Wenjie Wang, Yifei Feng, Bo Peng, Jianguo Yao, Haibing Guan  
**Institution**: Shanghai Jiao Tong University  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/liu-yao)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">mixed-precision</span>
<span class="paper-tag">kernel-selection</span>
<span class="paper-tag">edge-inference</span>
</div>

## Background

W4A8 and other asymmetric formats lack native hardware operations. Padding, LUT, and bit-disaggregation each win for different shapes, bit widths, and prefill/decode phases; one static method is suboptimal.

## Methodology

The Decomposition-Partial Product-Reconstruction model systematically generates mixed-precision algorithms. ADAngel builds optimized kernel strategies, exhaustively profiles an oracle map for a target model/hardware, and dispatches each runtime GEMM/GEMV shape to its best strategy.

## Experiment

Decode throughput is up to 5.10x llama.cpp; prefill TTFT improves 1.17-2.38x over TensorRT-LLM across evaluated arbitrary-precision configurations.

## Limitation

- Specialization/profile cost repeats for each model/hardware target.
- Strategy coverage bounds achievable performance.
- Quantization quality remains outside the runtime optimizer.

---

*Reading date: 2026-08*
*Note status: Completed*
