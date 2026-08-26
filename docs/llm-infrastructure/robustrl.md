# RobustRL: Role-Based Fault Tolerance for RL Post-Training

<div class="paper-meta" markdown>

**Authors**: Zhenqian Chen, Baoquan Zhong, Xiang Li, Qing Dai, Xinkui Zhao, Miao Ye, Ren Cheng, Lufei Zhang, Jianwei Yin  
**Institution**: Zhejiang University; other collaborators  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/chen-zhenqian)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">fault-tolerance</span>
<span class="paper-tag">rl-post-training</span>
<span class="paper-tag">role-isolation</span>
</div>

## Background

RL combines trainer, rollout, and management roles. Pretraining recovery restarts the whole job, wasting valid trajectories and initialized inference state even when only one role failed.

## Methodology

RobustRL uses Detect-Restart-Reconnect: role-aware monitors distinguish failure from normal role behavior; only the failed trainer/rollout role restarts, warm rollouts help restore trainers, and dynamic UCX point-to-point connections replace static collectives for reconnection.

## Experiment

On a 256-GPU Qwen3-8B-Math job with 10% injected failures, RobustRL exceeds 80% ETTR versus 60% for ByteRobust and shortens end-to-end training by 8.4-17.4%.

## Limitation

- Role state and algorithm semantics must tolerate partial restart.
- Dynamic point-to-point synchronization is more complex than static collectives.
- Evaluation focuses on machine errors; correlated cluster failures remain difficult.

---

*Reading date: 2026-08*
*Note status: Completed*
