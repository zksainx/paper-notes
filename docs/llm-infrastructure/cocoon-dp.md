# Cocoon: A System Architecture for Differentially Private Training with Correlated Noises

<div class="paper-meta" markdown>

**Authors**: Donghwan Kim, Xin Gu, Jinho Baek, Timothy Lo, Younghoon Min, Kwangsik Shin, Jongryool Kim, Jongse Park, Kiwan Maeng  
**Institution**: Penn State; SK hynix; KAIST  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/kim-donghwan)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">differential-privacy</span>
<span class="paper-tag">training</span>
<span class="paper-tag">near-memory-processing</span>
</div>

## Background

Correlated-noise DP improves accuracy by canceling noise across iterations, but requires a large noise history. Large models and embedding tables turn generation/storage/application of this history into the bottleneck.

## Methodology

Cocoon tiers noise state across GPU, CPU, and memory-extension capacity, specializes sparse-embedding processing, and offloads suitable operations to a near-memory-processing device. Placement follows capacity and bandwidth rather than forcing all privacy state into HBM.

## Experiment

On a real platform with an FPGA NMP prototype, Cocoon improves correlated-noise training performance by 1.23-10.82x across evaluated models/mechanisms.

## Limitation

- Relies partly on not-yet-commodity NMP hardware.
- It accelerates a chosen DP mechanism; privacy/utility parameters remain an algorithmic decision.
- Tiering very large noise history adds failure and consistency concerns.

---

*Reading date: 2026-08*
*Note status: Completed*
