# Teaching the Old Dog New Tricks: Efficient Data Pipelines for Large-Scale LLM Pre-Training

<div class="paper-meta" markdown>

**Authors**: Luofan Chen, Chenhan Wang, Weidong Zhang, Jinxin Chi, Hequan Zhang, et al.  
**Institution**: USTC; ByteDance Seed; ByteDance; Tsinghua University  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/chen-luofan)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">data-pipeline</span>
<span class="paper-tag">checkpoint-io</span>
<span class="paper-tag">multimodal-training</span>
</div>

## Background

Thirty thousand production traces over 90 days expose three storage mismatches: cross-DC evaluation performs thousands of latency-bound checkpoint reads, synchronized startup creates hot-file storms, and multimodal decoding/transformation exhausts training-host CPUs.

## Methodology

| Bottleneck | Optimization |
| --- | --- |
| Cross-DC evaluation | Predict checkpoint demand from a global namespace and replicate proactively |
| Startup I/O storm | Replicate hot metadata/parameter files before synchronized reads |
| Transformation wall | Offload image/video transformation to underused storage-tier CPUs |

## Experiment

Wasted GPU hours per evaluation fall from 16,800 to 4,000; checkpoint startup loading improves by 40.8%; data-loading stalls fall by 63.2%. The design upgrades an HDFS-based exabyte-scale substrate without replacing it.

## Limitation

- Predictability assumptions weaken for highly exploratory data selection.
- Replication consumes storage and cross-DC bandwidth.
- Transformation offload shifts load to shared storage CPUs and can create new contention.

---

*Reading date: 2026-08*
*Note status: Completed*
