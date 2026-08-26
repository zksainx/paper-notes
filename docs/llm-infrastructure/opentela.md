# OpenTela: Unifying Decentralized Computing Resources for Heterogeneous LLM Serving

<div class="paper-meta" markdown>

**Authors**: Xiaozhe Yao, Youhe Jiang, Ilia Badanin, Qinghao Hu, Robert Matthew Smith, Binhang Yuan, Imanol Schlag, Eiko Yoneki, Ana Klimovic  
**Institution**: ETH Zurich; Cambridge; EPFL; MIT; HKUST  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/yao)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">sovereign-ai</span>
<span class="paper-tag">hpc</span>
<span class="paper-tag">control-plane</span>
</div>

## Background

Slurm-managed HPC allocations are transient, externally unreachable, and fragmented across institutions; vLLM/SGLang assume a persistent Kubernetes-like control plane.

## Methodology

OpenTela is a rootless user-space overlay: CRDT gossip provides fault-tolerant discovery, adapters unify schedulers/engines, gateways preserve endpoints, and a heterogeneity-aware scheduler places models across decentralized clusters.

## Experiment

Over 22 months it served 13 million requests and 15 billion tokens across 142 models for more than 1,000 researchers. The production trace and system are released; latency simulation is within 10% and heterogeneous speedup prediction within 6% in reported validation.

## Limitation

- Wide-area failures/security policies remain institutional constraints.
- Gossip convergence is eventual, so routing state can be temporarily stale.
- User-space deployment cannot fix all scheduler/network limitations.

---

*Reading date: 2026-08*
*Note status: Completed*
