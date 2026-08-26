# LLM Infrastructure at OSDI/SOSP (2025-2026)

This collection covers papers in the official OSDI 2025, OSDI 2026, SOSP 2025, and SOSP 2026 programs whose primary system object is large-model training, inference, serving, model/activation storage, agent runtime, or production reliability. It intentionally keeps one category and four venue-year notes rather than creating a directory per subtopic.

## Scope

- Included: LLM/foundation-model training and post-training, inference and serving, KV/context/weight storage, agentic workflows, GPU/runtime orchestration, and production correctness/diagnostics.
- Excluded: generic GPU compilers, ordinary databases, vector search, or papers that only use an LLM as a developer tool when the LLM is not the system being operated.
- Evidence: official conference abstracts and paper pages; SOSP 2026 preprint abstracts are linked when available. A title-only entry is marked as such instead of inventing quantitative claims.

## Coverage

| Venue | Coverage | Papers |
| --- | --- | ---: |
| OSDI 2025 | Wafer-scale inference, autoscaling, training balance, serving throughput, low-bit inference | 6 |
| SOSP 2025 | Training reliability, KV/memory management, serving, RAG, hybrid inference | 16 |
| OSDI 2026 | Long-context serving, MoE/RL training, data pipelines, observability, SLOs, fault tolerance | 30 |
| SOSP 2026 | Training emulation, production serving, agent runtimes, model storage, verification | 18 |

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

Five accepted SOSP 2026 papers had no public camera-ready text as of 2026-08-26: ACDC, Beyond Utilization, MeshRT, SANDHI, and SystemX. They are excluded from the required note set by explicit user decision; see [Source Status](source-status.md) for the evidence trail.

## Synthesis

- [Research Landscape](research-landscape.md) - macro-level taxonomy and cross-paper conclusions.
- [Source Status](source-status.md) - per-venue coverage and unpublished-paper audit.

## Cross-Year Synthesis

- **Serving is becoming memory- and state-centric.** KV caching, weight placement, GPU ballooning, CPU/GPU hybrids, and disaggregated I/O dominate the 2025-2026 systems agenda.
- **Training systems are moving from static parallelism to adaptive control.** Workload-aware sharding, heterogeneous MoE pipelines, RL post-training co-scheduling, hot switching, and interruption recovery all treat the runtime as a feedback loop.
- **Reliability is now a first-class LLM systems problem.** Silent data corruption, deterministic inference, tracing, equivalence checking, and production diagnosis appear alongside throughput work.
- **Agents change the unit of orchestration.** Agent workflows, streaming-data forks, voice context orchestration, and cloud control planes require state isolation and policy-aware scheduling rather than a single request queue.

---

*Reading date: 2026-08*
*Note status: Completed*
