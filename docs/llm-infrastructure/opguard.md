# OpGuard: Bitwise Alignment for Precise and General Debugging of Production LLM Training

<div class="paper-meta" markdown>

**Authors**: Ziming Zhou, Yinjie Zhao, Hang Zhu, Wenxiao Wang, Zhihao Bai, Yun Zhang, Shuguang Wang, Haibin Lin, Peng Huang  
**Institution**: University of Michigan; ByteDance Seed  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/zhou-ziming)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">debugging</span>
<span class="paper-tag">bitwise-alignment</span>
<span class="paper-tag">training-correctness</span>
</div>

## Background

Loss/gradient curves aggregate millions of operations and reveal corruption long after the origin. Comparing two runs is useful only if benign nondeterminism can be removed and corresponding operator states aligned.

## Methodology

OpGuard finds semantic-stable operator boundaries across heterogeneous stacks, fingerprints tensors, maps schedules to find the longest bitwise-identical prefix, and presents the first mismatch with rank/operator/context. Controlled determinism makes that pivot strong evidence.

## Experiment

Deployed across ByteDance pre/post-training, OpGuard diagnosed more than 20 production issues, including kernel races and SDCs, reducing representative debugging cycles from days to minutes.

## Limitation

- Requires a reference execution and repeatable inputs.
- Enforcing determinism can perturb performance/ordering.
- Bugs before the aligned region or shared by both runs can evade differential diagnosis.

---

*Reading date: 2026-08*
*Note status: Completed*
