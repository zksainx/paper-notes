# SDCs in the Wild: Characterizing and Diagnosing SDC-Defective GPUs in Production LLM Training

<div class="paper-meta" markdown>

**Authors**: Wenxin Zheng, Wenxiao Wang, Yun Zhang, Mingcong Han, Bin Xu, et al.  
**Institution**: Shanghai Jiao Tong University; ByteDance Seed  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/zheng)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">silent-data-corruption</span>
<span class="paper-tag">execution-replay</span>
<span class="paper-tag">gpu-diagnosis</span>
</div>

## Background

Synthetic GEMM/stress tests miss over 60% of defective GPUs because logic-level SDCs are aged, data-dependent, unit-specific, and invisible to ECC/thermal protection. Their symptoms resemble software bugs.

## Methodology

SDCHunter captures the exact workload/input that triggered an incident and performs deterministic hierarchical replay. Coarse DP/PP/EP tensor signatures isolate a group with low overhead; finer replay then identifies the device, followed by vendor/offline confirmation.

## Experiment

The characterization covers 23 confirmed SDC-defective GPUs. SDCHunter mitigated 40 production incidents; coarse signature instrumentation is reported around 3-4.3% overhead, versus 683.9% for global fine-grained monitoring.

## Limitation

- Rare faults may take hours to reproduce even with faithful replay.
- Determinizing kernels/collectives changes the execution environment.
- The method diagnoses persistent defective devices better than transient cosmic-ray-like faults.

---

*Reading date: 2026-08*
*Note status: Completed*
