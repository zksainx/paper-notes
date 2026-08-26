# DecDEC: A Systems Approach to Advancing Low-Bit LLM Quantization

<div class="paper-meta" markdown>

**Authors**: Yeonhong Park, Jake Hyun, Hojoon Kim, Jae W. Lee  
**Institution**: Seoul National University  
**Conference**: OSDI '25  
**Year**: 2025  
**Paper Link**: [USENIX paper](https://www.usenix.org/conference/osdi25/presentation/park-yeonhong)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">quantization</span>
<span class="paper-tag">edge-inference</span>
<span class="paper-tag">memory-offload</span>
</div>

## Background

3-4 bit quantization makes large models fit on consumer and mobile GPUs, but quantization error is amplified in channels corresponding to activation outliers. Storing a full-precision copy defeats the memory benefit, while blindly fetching error corrections from CPU memory can erase the latency gain.

**Key Takeaways**

- DecDEC stores only the residual between full-precision and quantized weights in CPU memory.
- At each decode step it identifies salient channels from current activations and fetches corrections only for those channels.
- The runtime adapts to input-dependent saliency instead of relying on an offline calibration set.

## Methodology

| Stage | Operation |
| --- | --- |
| Quantized base | Keep the low-bit weights in GPU memory |
| Residual store | Keep full-minus-quantized corrections in CPU memory |
| Saliency detector | Inspect activation outliers during each decode step |
| Selective fetch | Transfer residuals for salient channels only |
| Corrected GEMV | Apply the residual contribution with low extra GPU memory |

The core systems trade-off is PCIe bandwidth versus quality recovery. Because only channels multiplied by large activation values produce large error, a small transfer can recover disproportionate perplexity loss.

## Experiment

| Setting | Result |
| --- | --- |
| 3-bit Llama-3-8B-Instruct | Perplexity reduced from 10.15 to 9.12, better than the 3.5-bit counterpart |
| GPU memory overhead | Less than 0.0003% in the highlighted RTX 4050 Mobile case |
| Inference latency | 1.7% slowdown in the same case |
| Other models | Evaluated with Llama-3, Phi-3, AWQ and multiple consumer GPU generations |

The benefit is strongest when GPU memory is the binding constraint and PCIe transfer is relatively cheap compared with the saved precision. On high-bandwidth server GPUs, GEMV or memory bandwidth can dominate and reduce the relative gain.

## Limitation

- Saliency detection and selective residual fetch add runtime complexity.
- Gains vary with GPU generation, PCIe bandwidth, model architecture, and quantizer.
- The method improves perplexity but does not remove the quality/bitwidth trade-off entirely.

---

*Reading date: 2026-08*
*Note status: Completed*
