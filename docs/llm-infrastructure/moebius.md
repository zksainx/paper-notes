# Moebius: Serving MoE Models with Seamless Runtime Parallelism Switching

<div class="paper-meta" markdown>

**Authors**: Yizhuo Liang, Shaoyu Wang, Jaeyong Song, Yanqi Zhou, Geon-Woo Kim, Guangrong He, Seo Jin Park  
**Institution**: University of Southern California; Google DeepMind; UT Austin  
**Conference**: SOSP '26  
**Year**: 2026  
**Paper Link**: [arXiv:2606.26607](https://arxiv.org/abs/2606.26607)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">moe-serving</span>
<span class="paper-tag">parallelism-switching</span>
<span class="paper-tag">rl-rollout</span>
</div>

## Background

Tensor parallelism wins at low concurrency; expert parallelism wins at high concurrency. Online bursts and RL rollouts cross this boundary during one live workload, but restarting or draining an engine to change layout is unacceptable.

## Methodology

Moebius recognizes TP/EP as two ownership layouts over byte-identical weights and KV. It keeps both runtimes resident, uses fixed virtual addresses and fused GPU transfers to move only slices whose owners change, updates requests/KV between decode steps, and cancels switches when live KV cannot fit.

## Experiment

On 8 H200 GPUs serving Qwen3-235B-A22B, Moebius beats the better static layout by 1.16-1.25x (1.22x mean) across RL rollout steps. Switching takes 215-434 ms with 2.4% memory overhead. Projected synchronous RL improvement is 1.10-1.20x when rollout occupies 65-85% of time.

## Limitation

- Requires high-bandwidth intra-node links and dual-mode runtime support.
- The measured result is generation-step speedup; end-to-end RL gains are projections.
- KV capacity can prevent switching exactly when long-tail contexts are largest.

---

*Reading date: 2026-08*
*Note status: Completed*
