# LLM Infrastructure at OSDI/SOSP (2025-2026)

This guide covers papers in the official OSDI 2025, OSDI 2026, SOSP 2025, and SOSP 2026 programs whose primary system object is large-model training, inference, serving, model/activation storage, agent runtime, or production reliability. It is the single entry point for coverage status, individual paper notes, and the cross-paper research landscape.

## Scope

- Included: LLM/foundation-model training and post-training, inference and serving, KV/context/weight storage, agentic workflows, GPU/runtime orchestration, and production correctness/diagnostics.
- Excluded: generic GPU compilers, ordinary databases, vector search, or papers that only use an LLM as a developer tool when the LLM is not the system being operated.
- Evidence: official conference/author PDFs, complete `arxiv2md` conversions, and official project artifacts when a camera-ready paper is not public. Title-only entries are not summarized.

## Coverage

| Venue | Accepted / in scope | Notes | Archived sources | Coverage |
| --- | ---: | ---: | ---: | --- |
| OSDI 2025 | 6 | 6 | 6 | Wafer inference, autoscaling, training balance, serving, quantization |
| SOSP 2025 | 16 | 16 | 15 | Reliability, KV/memory, serving, RAG, heterogeneous inference |
| OSDI 2026 | 30 | 30 | 30 | Long context, MoE/RL, data, observability, SLOs, fault tolerance |
| SOSP 2026 | 18 | 13 | 13 | Emulation, serving, agents, model storage, verification |
| **Total** | **70** | **65** | **64** | Complete for all publicly available sources |

Five SOSP 2026 papers had no public camera-ready text as of 2026-08-26 and were explicitly waived: **ACDC**, **Beyond Utilization**, **MeshRT**, **SANDHI**, and **SystemX**. ACDC exposes production traces but not its system design; the other four expose only title/author metadata. They are intentionally not reconstructed from titles.

Originals are archived under `originals/osdi25/`, `originals/sosp25/`, `originals/osdi26/`, and `originals/sosp26/`.

## OSDI 2025 Paper Notes

- [WaferLLM](waferllm.md) - wafer-scale parallelism and mesh-local inference.
- [BlitzScale](blitzscale.md) - live autoscaling with compute-network parameter loading.
- [Understanding Stragglers](understanding-stragglers.md) - counterfactual diagnosis of distributed training slowdowns.
- [NanoFlow](nanoflow.md) - intra-device overlap for serving throughput.
- [WLB-LLM](wlb-llm.md) - workload-aware 4D training balance.
- [DecDEC](decdec.md) - selective CPU residuals for low-bit inference.

## SOSP 2025 Paper Notes

- [ByteRobust](byterobust.md) - production fault isolation and recovery for 10K-GPU training.
- [Mycroft](mycroft.md) - collective-level dependency tracing and root-cause analysis.
- [DCP](dcp.md) - per-batch context parallelism through hypergraph partitioning.
- [TrainVerify](trainverify.md) - formal equivalence checking for distributed execution plans.
- [Pie](pie.md) - programmable, sandboxed serving for agentic workflows.
- [DiffKV](diffkv.md) - differentiated KV compression with on-GPU compaction.
- [Jenga](jenga.md) - heterogeneous embedding allocation and cache policies.
- [IC-Cache](ic-cache.md) - in-context examples as a cross-model serving cache.
- [KTransformers](ktransformers.md) - CPU/GPU hybrid inference for very large MoE models.
- [PrefillOnly](prefillonly.md) - specialized KV management and JCT scheduling for one-token outputs.
- [Mercury](mercury.md) - compiler search over multi-GPU operators and remote memory.
- [HedraRAG](hedrarag.md) - graph transformations for coordinated generation and retrieval.
- [HeteroInfer](heteroinfer.md) - concurrent GPU/NPU inference on mobile SoCs.
- [Aegaeon](aegaeon.md) - token-granular autoscaling for multi-model GPU pools.
- [Sailor](sailor.md) - planning across heterogeneous and geo-distributed training resources.
- [METIS](metis-rag.md) - per-query quality/latency adaptation for RAG.

## OSDI 2026 Paper Notes

- [Strata](strata.md) - hierarchical context caching with GPU-assisted I/O.
- [ECHO](echo-kv.md) - lossless KV prefetch for native sparse attention.
- [DirectKV](directkv.md) - zero-copy CPU-resident KV on NVLink-C2C systems.
- [LMETRIC](lmetric.md) - tuning-free multiplicative request routing.
- [Prism](prism-serving.md) - elastic GPU memory ballooning across models.
- [Tessera](tessera.md) - overlap-aware pipelines for heterogeneous trillion-parameter MoE training.
- [LLM Data Pipeline](llm-data-pipeline.md) - cross-DC checkpoints, startup storms, and multimodal transforms.
- [Cocoon](cocoon-dp.md) - tiered correlated-noise state for private training.
- [Kareus](kareus.md) - joint frequency and kernel scheduling for time/energy trade-offs.
- [Weave](weave-rl.md) - cross-job co-scheduling for synchronous disaggregated RL.
- [RLinf](rlinf.md) - macro-to-micro transformation of heterogeneous RL workflows.
- [DynaRL](dynarl.md) - online resource migration following RL bottlenecks.
- [RollArt](rollart.md) - trajectory-level disaggregation across heterogeneous hardware.
- [Seer](seer-rl.md) - context-aware balancing and speculative decoding for synchronous rollout.
- [SPEX](spex.md) - speculative branch exploration for Tree-of-Thought reasoning.
- [Murakkab](murakkab.md) - declarative cross-layer orchestration for agent workflows.
- [StriaTrace](striattrace.md) - low-overhead production inference diagnosis.
- [SDCHunter](sdchunter.md) - workload-faithful replay for defective GPU diagnosis.
- [AEGIS](aegis-sdc.md) - sensor/verifier online SDC detection at fleet scale.
- [OpGuard](opguard.md) - bitwise-aligned differential debugging for training.
- [RobustRL](robustrl.md) - role-isolated recovery for RL post-training.
- [Local MoE SLO](local-moe-slo.md) - cloud-style local inference for intact MoE models.
- [UEP](uep.md) - portable expert-parallel communication across GPU/NIC vendors.
- [BatchGen](batchgen.md) - sequence coroutines for fleet-scale batch inference.
- [EcoServe](ecoserve.md) - partial disaggregation for commodity GPU networks.
- [Pipeline Serving](pipeline-serving.md) - dynamic chunks and decode balancing for online PP.
- [OpenTela](opentela.md) - decentralized user-space serving across HPC institutions.
- [Kairox](kairox.md) - online hot-neuron movement in CPU/GPU inference.
- [ADAngel](adangel.md) - workload-adaptive mixed-precision kernel selection.
- [Sereno](sereno.md) - foreground-aware bandwidth yielding on mobile SoCs.

## SOSP 2026 Paper Notes

- [PrismLLM](prismllm.md) - target-scale training emulation with a few physical ranks.
- [LLM-42](llm-42.md) - per-request deterministic inference through verify/rollback.
- [Cerium](cerium.md) - compiler/runtime support for TB-scale encrypted inference.
- [Moebius](moebius.md) - live TP/EP switching as concurrency changes.
- [TensorHub](tensorhub.md) - copy-free reference storage for RL weight versions.
- [YoloFS](yolofs.md) - staged, undoable, permission-aware files for coding agents.
- [Model2Kernel](model2kernel.md) - model-constrained symbolic execution of CUDA kernels.
- [SkVM](skvm.md) - capability-based compilation of portable agent skills.
- [AgileLog](agilelog.md) - continuous forks for isolated agents on live streams.
- [TStore](tstore.md) - content-driven tensor-level model delta compression.
- [Janus](janus-serving.md) - dual-timescale multi-LLM production serving.
- [LMTracer](lmtracer.md) - graph-embedded low-overhead GPU profiling.
- [Voice Context](voice-context.md) - structured long-running context for voice LLMs.

## Research Landscape

The dominant problem across these papers is no longer simply faster matrix multiplication. It is **state management under dynamic execution**: deciding where weights, KV cache, optimizer tensors, context, agent side effects, communication dependencies, privacy noise, and model versions live; when they move; and which invariants make those transitions safe.

Three transitions organize the literature:

1. **Static placement to adaptive control.** Parallelism, memory allocation, routing, precision, and hardware assignment increasingly change online.
2. **Single-request engines to workflow runtimes.** RAG, agents, RL, voice, and Tree-of-Thought expose multi-stage dependencies that a token loop cannot optimize globally.
3. **Best-effort speed to production contracts.** Determinism, SDC detection, formal equivalence, fault isolation, energy, privacy, and side-effect control become first-class objectives.

### Research Map

| State being managed | Representative systems | Core control action |
| --- | --- | --- |
| Weights and versions | BlitzScale, Prism, Aegaeon, Janus, TensorHub, TStore | Load, balloon, fork, transfer, compress, retain |
| KV and context | DiffKV, Jenga, Strata, ECHO, DirectKV, PrefillOnly, Voice Context | Compress, tier, prefetch, drop, project |
| Parallel execution | WLB-LLM, DCP, Tessera, Mercury, Moebius, UEP | Repartition, switch, overlap, rebalance |
| RL workflow state | Weave, RLinf, DynaRL, RollArt, Seer, RobustRL | Co-schedule, transform, migrate, isolate roles |
| Correctness evidence | TrainVerify, Mycroft, OpGuard, SDCHunter, AEGIS, LLM-42, Model2Kernel | Trace, replay, verify, compare, rollback |
| Agent side effects | Pie, Murakkab, YoloFS, AgileLog, SkVM | Program, sandbox, fork, gate, compile |
| Hardware resources | WaferLLM, HeteroInfer, KTransformers, Kairox, ADAngel, Sereno | Map to topology, CPU/GPU/NPU, precision, bandwidth |

### Memory as the Serving Control Plane

Serving papers increasingly use memory allocation as scheduling. Aegaeon moves model state at token granularity; Prism unifies spatial and temporal sharing through ballooning; Janus separates steady and burst pools over a virtual HBM substrate. Strata, ECHO, and DirectKV turn CPU/SSD memory into scheduled cache tiers rather than passive overflow.

Request routing and memory management are converging: the best destination depends on which state is present, how cheaply missing state can arrive, and what eviction externality a request creates.

### Parallelism as a Runtime Variable

WLB-LLM balances document-dependent attention work, DCP repartitions each batch, Tessera uses post-overlap cost, Moebius switches TP and EP between decode steps, and pipeline serving changes chunk size from live SLO slack. Parallelism is becoming a stateful online decision rather than a launch-time configuration.

The main systems challenge shifts from finding one optimal plan to defining safe transitions between plans while preserving weights, KV state, communication groups, and in-flight requests.

### RL Post-Training as a Distinct Systems Domain

RL combines memory-bound rollout, compute-bound optimization, tool/environment latency, bursty reward work, and frequent weight publication. Weave fills cross-job bubbles; RLinf transforms workflows into micro-flows; DynaRL follows moving bottlenecks; RollArt maps stages to heterogeneous hardware; Seer handles synchronous rollout tails; TensorHub distributes versions; RobustRL recovers roles independently.

The unresolved problem is jointly optimizing throughput and learning semantics: staleness, sampling bias, partial trajectories, and recovery affect the learned policy, not only system performance.

### Reliability Near the First Divergence

Loss curves are late and ambiguous. Mycroft traces collective dependencies; OpGuard locates the first bitwise mismatch; SDCHunter replays the triggering workload; AEGIS escalates from cheap sensing to verification; TrainVerify proves plans before execution; Model2Kernel constrains CUDA symbolic execution with model semantics.

The common pattern is a cheap structured invariant on the fast path, followed by focused escalation near suspicious evidence. This scales better than continuous full tracing or duplicated execution.

### Reversible State for Agents

Pie makes inference programmable, Murakkab makes workflows declarative, and SkVM compiles skills for model/harness capabilities. YoloFS stages and snapshots mutations; AgileLog gives agents isolated continuous forks of live streams.

The emerging primitive is a **reversible branch of state**. Agent autonomy becomes practical when exploration is cheap, isolated, inspectable, and promotable. This idea can extend from files and logs to databases, cloud resources, credentials, and external tool transactions.

### Hardware Heterogeneity as an Input

WaferLLM replaces GPU assumptions with PLMR; HeteroInfer schedules GPU and NPU together; KTransformers and Kairox use CPU compute and DRAM; UEP restores GPU/NIC portability through CPU proxies; ADAngel selects mixed-precision algorithms by workload; Sereno yields NPU bandwidth to foreground apps.

These systems expose meaningful hardware asymmetry to planners while retaining a stable higher-level contract, rather than hiding every device behind an identical abstraction.

### Open Problems

- **Control-loop composability:** cache, routing, frequency, and parallelism controllers can oscillate or fight each other.
- **State-transition correctness:** live migration needs transactional guarantees across weights, KV, requests, and communication groups.
- **Quality-aware metrics:** throughput and latency do not capture RAG quality, speculative waste, quantization error, RL staleness, or agent outcomes.
- **Reproducibility:** production results often depend on private traces, specialized hardware, and changing model versions.
- **Programmable-system security:** inferlets, skills, agent forks, and adaptive control planes expand the trusted computing base.
- **Energy accounting:** few systems jointly optimize energy, embodied hardware cost, and SLOs.

The unifying direction is **adaptive, state-aware infrastructure with explicit correctness boundaries**. The strongest systems expose hidden state, attach an invariant or cost to it, and safely change its placement or execution policy at finer granularity than previous stacks.

---

*Reading date: 2026-08*
*Note status: Completed for all publicly available sources*
