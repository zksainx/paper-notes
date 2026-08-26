# Sereno: Taming Memory Bandwidth Contention in Mobile LLM Inference

<div class="paper-meta" markdown>

**Authors**: Tong Xin, Xinrui Shi, Mingkai Dong, Zeyu Mi  
**Institution**: Shanghai Jiao Tong University  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/xin)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">mobile-inference</span>
<span class="paper-tag">memory-qos</span>
<span class="paper-tag">speculative-decoding</span>
</div>

## Background

Mobile NPUs inherit high-priority DRAM arbitration intended for real-time media. Background LLM inference therefore raises foreground jank by 153%, while its own prefill/decode loses only 1.01%/1.64%.

## Methodology

Sereno repurposes speculative decoding to create fine-grained, progress-preserving yield points. Runtime contention detection pauses/yields NPU memory bandwidth to foreground rendering and resumes without discarding generated progress.

## Experiment

Across commercial phones and 25 apps, foreground jank falls up to 92.6% (58.5% average) while LLM throughput rises up to 67.9% (26.4% average). Versus vanilla speculative decoding, jank falls up to 72.1% with 6.2% inference degradation.

## Limitation

- Depends on mobile NPU/runtime preemption and traffic behavior.
- Speculative decoding needs a suitable draft/verification setup.
- QoS policy must decide how much background progress to sacrifice.

---

*Reading date: 2026-08*
*Note status: Completed*
