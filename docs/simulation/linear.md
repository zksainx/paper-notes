# Path Forward Beyond Simulators: Fast and Accurate GPU Execution Time Prediction for DNN Workloads

<div class="paper-meta" markdown>

**Authors**: Ying Li, Yifan Sun, Adwait Jog  
**Institution**: Multiple Institutions  
**Conference**: MICRO 2023  
**Paper Link**: [ACM DL](https://dl.acm.org/doi/10.1145/3613424.3614277)

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">GPU Performance</span>
<span class="paper-tag">DNN Prediction</span>
<span class="paper-tag">Linear Regression</span>
</div>

## Abstract

This paper proposes a fast, linear-regression-based execution time predictor for DNNs as an alternative to slow simulators. It predicts new DNN performance with 7% error and new GPU performance with 15.2% error, offering a practical solution for modeling large-scale systems and DNNs.

**Key Contributions**:
- Fast linear-regression-based DNN execution time predictor
- Multiple performance models: End-to-End (E2E), Layer-Wise (LW), Kernel-Wise (KW), and Inter-GPU Kernel-Wise (IGKW)
- Large-scale dataset with 646 models from HuggingFace, TorchVision
- Practical alternative to time-consuming simulators

## Core Ideas

Simulators are becoming increasingly impractical for modeling today's large-scale systems and DNNs due to long simulation times. This work collects execution time data for DNNs, layers, and kernels on various GPUs, then develops lightweight regression models that maintain acceptable accuracy while dramatically reducing prediction time compared to full simulation approaches.

---

*Reading date: 2025*
*Note status: Completed*
