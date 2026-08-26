# LLM Infrastructure Research Landscape: OSDI/SOSP 2025-2026

## Executive View

The 2025-2026 papers show LLM infrastructure moving beyond “make matrix multiplication faster.” The dominant problem is now **state management under dynamic execution**. The relevant state may be weights, KV cache, optimizer tensors, context, agent side effects, communication dependencies, privacy noise, or model versions. Performance comes from changing where that state lives, when it moves, who owns it, and how precisely the runtime can observe it.

Three broad transitions organize the literature:

1. **Static placement to adaptive control.** Parallelism, memory allocation, routing, precision, and hardware assignment increasingly change online.
2. **Single-request engines to workflow runtimes.** RAG, agents, RL, voice, and Tree-of-Thought expose multi-stage dependencies that a token loop cannot optimize globally.
3. **Best-effort speed to production contracts.** Determinism, SDC detection, formal equivalence, fault isolation, energy, privacy, and user-side-effect control become first-class system objectives.

## Research Map

| State being managed | Representative systems | Core control action |
| --- | --- | --- |
| Weights/model versions | BlitzScale, Prism, Aegaeon, Janus, TensorHub, TStore | Load, balloon, fork, transfer, compress, retain |
| KV/context | DiffKV, Jenga, Strata, ECHO, DirectKV, PrefillOnly, Voice Context | Compress, tier, prefetch, drop, project |
| Parallel execution | WLB-LLM, DCP, Tessera, Mercury, Moebius, UEP | Repartition, switch, overlap, rebalance |
| RL workflow state | Weave, RLinf, DynaRL, RollArt, Seer, RobustRL | Co-schedule, transform, migrate, isolate roles |
| Correctness evidence | TrainVerify, Mycroft, OpGuard, SDCHunter, AEGIS, LLM-42, Model2Kernel | Trace, replay, verify, compare, rollback |
| Agent side effects | Pie, Murakkab, YoloFS, AgileLog, SkVM | Program, sandbox, fork, gate, compile |
| Hardware resources | WaferLLM, HeteroInfer, KTransformers, Kairox, ADAngel, Sereno | Map to topology, CPU/GPU/NPU, precision, bandwidth |

## Direction 1: Memory Is Becoming the Serving Control Plane

Serving papers increasingly make memory allocation the mechanism for scheduling. Aegaeon scales at token granularity by moving model state. Prism unifies spatial and temporal sharing through ballooning. Janus separates a steady colocated pool from a burst pool and gives both a shared virtual HBM substrate. Strata, ECHO, and DirectKV treat CPU/SSD memory not as passive overflow but as a scheduled cache tier with GPU-aware access.

This suggests that future serving engines will expose a common state API spanning weights, KV, adapters, activations, and prefix objects. Request routing and memory management will converge: the best destination is determined by which state is already present, how cheaply missing state can arrive, and what eviction externality the request creates.

## Direction 2: Parallelism Is a Runtime Variable

Earlier systems selected TP/PP/DP/EP before execution. The new papers repeatedly invalidate that assumption. WLB-LLM balances document-dependent attention work; DCP repartitions every batch; Tessera partitions according to post-overlap cost; Moebius switches TP and EP between decode steps; pipeline serving adjusts chunk size from live SLO slack.

The deeper lesson is that a parallelism degree is not a configuration knob but a **stateful online decision**. Switching cost depends on weight/KV ownership, communication topology, and in-flight requests. Research therefore shifts from finding one optimal plan to defining safe transition mechanisms between plans.

## Direction 3: RL Post-Training Is Becoming Its Own Systems Domain

RL workloads are not ordinary training. They combine memory-bound rollout, compute-bound optimization, tool/environment latency, bursty reward work, and frequent weight publication. Weave fills one job's dependency bubbles with another. RLinf transforms logical workflows into micro-flows. DynaRL follows moving bottlenecks. RollArt maps stages to heterogeneous hardware and decouples trajectories. Seer targets synchronous long-tail rollout. TensorHub turns existing replicas into a weight-version store. RobustRL recovers roles independently.

The common abstraction is a versioned dataflow with heterogeneous actors. The remaining open problem is how to co-optimize throughput and learning semantics: staleness, sampling bias, partial trajectories, and failure recovery affect not only performance but the policy being learned.

## Direction 4: Reliability Moves Closer to the First Divergence

Aggregate loss curves are too late and too ambiguous. Mycroft traces collective dependencies; OpGuard finds the first bitwise operator mismatch; SDCHunter replays the exact triggering workload; AEGIS separates cheap sensing from definitive verification; TrainVerify proves distributed plans equivalent before execution; Model2Kernel constrains symbolic CUDA analysis with real model semantics.

These systems share a pattern: capture a cheap, structured invariant on the fast path, then escalate only near suspicious evidence. This is more scalable than continuous full tracing or full duplicated execution. Future stacks will likely combine these layers into a correctness pipeline from execution plan, through kernels/collectives, to fleet hardware and model output.

## Direction 5: Agent Infrastructure Requires Reversible State

Agents make decisions that cannot be trusted individually yet must run without constant human approval. Pie makes the inference loop programmable; Murakkab makes the whole workflow declarative; SkVM compiles skills for model/harness capabilities. YoloFS stages and snapshots file mutations; AgileLog gives agents isolated continuous forks of live streams.

The key systems primitive is not “an agent process” but a **reversible branch of state**. Agent autonomy becomes practical when exploration is cheap, isolated, inspectable, and promotable. This idea should generalize from files/logs to databases, cloud resources, credentials, and external tool transactions.

## Direction 6: Hardware Heterogeneity Is Used, Not Hidden

WaferLLM replaces GPU assumptions with the PLMR model. HeteroInfer schedules GPU and NPU concurrently. KTransformers and Kairox exploit CPU DRAM/compute rather than treating offload as a necessary evil. UEP inserts CPU proxies to regain GPU/NIC portability. ADAngel selects among asymmetric-precision algorithms per shape. Sereno explicitly yields NPU memory bandwidth to foreground applications.

This reverses the traditional portability goal. Instead of presenting identical hardware, systems expose meaningful asymmetry to a planner while preserving a stable higher-level contract.

## Open Problems

- **Control-loop composability:** independent adaptive schedulers for cache, routing, frequency, and parallelism can oscillate or fight each other.
- **State transition correctness:** live migration/switching needs transactional guarantees across weights, KV, requests, and communication groups.
- **Quality-aware systems metrics:** throughput and latency are insufficient for RAG, speculation, quantization, RL staleness, and agent outcomes.
- **Cross-layer reproducibility:** production results rely on private traces, specialized hardware, and changing model versions.
- **Security of programmable infrastructure:** inferlets, skills, agent forks, and adaptive control planes expand the trusted computing base.
- **Energy and capacity accounting:** only a few papers jointly optimize energy, embodied hardware cost, and SLOs.

## Bottom Line

The unifying research direction is **adaptive, state-aware infrastructure with explicit correctness boundaries**. The winning systems do not merely execute a fixed model faster. They expose hidden state, attach an invariant or cost to it, and safely change its placement or execution policy at a much finer granularity than previous stacks.

---

*Reading date: 2026-08*
*Note status: Completed for all publicly available sources*
