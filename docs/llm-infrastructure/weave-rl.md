# Weave: Efficient Co-Scheduling for Disaggregated RL Post-Training

<div class="paper-meta" markdown>

**Authors**: Tianyuan Wu, Lunxi Cao, Yining Wei, Wei Gao, Yuheng Zhao, Dakai An, et al.  
**Institution**: HKUST; UIUC; Alibaba Group  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/wu-tianyuan)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">rl-post-training</span>
<span class="paper-tag">co-scheduling</span>
<span class="paper-tag">disaggregation</span>
</div>

## Background

Disaggregating memory-bound rollout onto H20-like GPUs and compute-bound training onto H800-like GPUs improves fit, but synchronous on-policy dependencies alternate idle periods between pools.

## Methodology

Weave forms co-execution groups: one job's rollout fills another job's training bubble. An inter-group stochastic planner places jobs conservatively; an intra-group round-robin schedule is provably optimal under the model. Host-memory residency keeps model switches warm.

## Experiment

On 328 H20 plus 328 H800 GPUs, Weave improves cost efficiency by 1.84x over standard disaggregation and 1.38x over co-located baselines with 100% SLO attainment.

## Limitation

- Needs multiple concurrent jobs with complementary phases.
- Host residency constrains group membership and model count.
- The optimality result depends on the scheduling model and predictable iteration structure.

---

*Reading date: 2026-08*
*Note status: Completed*
