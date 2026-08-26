# Seer: Online Context Learning for Fast Synchronous LLM Reinforcement Learning

<div class="paper-meta" markdown>

**Authors**: Ruoyu Qin, Weiran He, Weixiao Huang, Yangkun Zhang, Yikai Zhao, Bo Pang, Xinran Xu, Yingdi Shan, Yongwei Wu, Mingxing Zhang  
**Institution**: Moonshot AI; Tsinghua University  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/qin)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">synchronous-rl</span>
<span class="paper-tag">speculative-decoding</span>
<span class="paper-tag">tail-latency</span>
</div>

## Background

Rollout consumes 63-87% of iteration time in reported production workloads. Output lengths are heavy-tailed, but requests sharing a prompt have correlated length and response structure. Asynchronous fixes bias data toward short trajectories and relax on-policy semantics.

## Methodology

| Technique | Role |
| --- | --- |
| Divided rollout | Redistribute unfinished requests dynamically across instances |
| Context-aware scheduling | Use same-prompt history to predict/migrate long-tail work |
| Grouped speculative decoding | Share context-derived draft behavior and adapt group/draft choices |

## Experiment

Seer improves synchronous rollout throughput by up to 2.04x and reduces long-tail latency by 72-94%. Grouped speculative decoding provides up to 1.3x over vanilla speculative decoding while reported training quality tracks synchronous RL.

## Limitation

- Relies on repeated prompts or contexts with predictive value.
- KV migration and re-prefill can consume the expected balancing gain.
- Speculative acceptance varies as the policy evolves.

---

*Reading date: 2026-08*
*Note status: Completed*
