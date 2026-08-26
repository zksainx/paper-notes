# TrainVerify: Equivalence-Based Verification for Distributed LLM Training

<div class="paper-meta" markdown>

**Authors**: Yunchi Lu, Youshan Miao, Cheng Tan, Peng Huang, Yi Zhu, Xian Zhang, Fan Yang  
**Institution**: University of Michigan; Microsoft Research; Northeastern University  
**Conference**: SOSP '25  
**Year**: 2025  
**Paper Link**: [arXiv:2506.15961](https://arxiv.org/abs/2506.15961)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">formal-verification</span>
<span class="paper-tag">distributed-training</span>
<span class="paper-tag">correctness</span>
</div>

## Background

Distributed parallelization can silently change gradients, aggregation, slicing, or rank assignment while the job continues to train. Differential testing is expensive at frontier scale and cannot prove correctness for all inputs. TrainVerify treats the single-device logical model as the specification and verifies the generated distributed execution plan before spending GPU time.

## Methodology

| Component | Role |
| --- | --- |
| Symbolic DFG | Represents logical and parallel execution algebraically |
| Tensor lineage | Maps distributed tensor slices back to logical tensors |
| Shape reduction | Proves representative small shapes whose structure generalizes |
| Stage-wise verification | Divides the plan by pipeline stages and verifies stages concurrently |
| SMT checking | Proves input/output equivalence or produces a counterexample |

The system integrates with nnScaler. It verifies execution plans, not arbitrary runtime code: the model definition is assumed correct, and crashes before plan generation remain outside scope.

## Experiment

| Setting | Result |
| --- | --- |
| Frontier plans | Successfully verifies Llama3-405B and DeepSeek-V3-671B plans |
| Parallel verification | Removing stage parallelism increases one case from under 18 s to over 90 s |
| Bug coverage | Detects incorrect transformations, missing/wrong synchronization, rank assignment, slicing, and aggregation errors |
| Large plans | Reported verification completes within practical pre-training setup time, typically seconds to under half an hour depending on plan |

## Limitation

- Correctness is relative to the logical model; a wrong specification is still accepted.
- Buffer races, framework crashes, floating-point nondeterminism, and hardware faults are not generally covered.
- Shape-reduction soundness and supported operator semantics form the trusted base.

---

*Reading date: 2026-08*
*Note status: Completed*
