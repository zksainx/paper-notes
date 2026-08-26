# Breaking the Reward Barrier: Accelerating Tree-of-Thought Reasoning via Speculative Exploration

<div class="paper-meta" markdown>

**Authors**: Shuzhang Zhong, Haochen Huang, Shengxuan Qiu, Pengfei Zuo, Runsheng Wang, Meng Li  
**Institution**: Peking University; ByteDance Seed  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/zhong)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">tree-of-thought</span>
<span class="paper-tag">speculation</span>
<span class="paper-tag">reasoning-inference</span>
</div>

## Background

Tree-of-Thought search waits for a reward before choosing the next branch, creating a reward dependency barrier that linear CoT serving optimizations cannot overlap.

## Methodology

SPEX speculatively selects promising branches within a query, allocates speculative budgets across queries, and terminates redundant/deep branches adaptively. It composes with token-level speculative decoding because branch and token speculation target different barriers.

## Experiment

Implemented on SGLang, SPEX gives 1.2-3x speedups across evaluated ToT algorithms and models while preserving the search result under its verification/reward policy.

## Limitation

- Wrong branch predictions waste generation compute.
- Early termination can change search quality if thresholds are aggressive.
- Benefits depend on reward latency and available parallel capacity.

---

*Reading date: 2026-08*
*Note status: Completed*
