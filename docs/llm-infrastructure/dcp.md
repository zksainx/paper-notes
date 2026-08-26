# DCP: Addressing Input Dynamism in Long-Context Training via Dynamic Context Parallelism

<div class="paper-meta" markdown>

**Authors**: Chenyu Jiang, Zhenkun Cai, Ye Tian, Zhen Jia, Yida Wang, Chuan Wu  
**Institution**: The University of Hong Kong; Amazon Web Services  
**Conference**: SOSP '25  
**Year**: 2025  
**Paper Link**: [arXiv:2510.10620](https://arxiv.org/abs/2510.10620)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">context-parallelism</span>
<span class="paper-tag">long-context</span>
<span class="paper-tag">hypergraph-partitioning</span>
</div>

## Background

Static context parallelism partitions every sequence uniformly even though sequence lengths and attention masks vary per batch. Short sequences pay unnecessary KV exchange; sparse masks communicate and compute blocks that will never interact. DCP chooses a new placement for each batch.

## Methodology

| Stage | Mechanism |
| --- | --- |
| Block construction | Partition Q, KV, output, and attention computation into fine-grained blocks |
| Hypergraph model | Vertices represent data/compute cost; hyperedges represent required communication |
| Placement | Balance memory/compute while minimizing weighted hyperedge cuts |
| Executor | Run blockwise attention, reduction, and copy kernels from the generated plan |

The balanced hypergraph problem is NP-hard, so DCP uses KaHyPar and amortizes planning through the data loader. A complete short sequence may stay on one GPU, while long/sparse sequences distribute only the blocks required by their mask.

## Experiment

| Workload | Result |
| --- | --- |
| Causal attention microbenchmarks | 1.19x-2.45x speedup |
| Sparse attention microbenchmarks | 2.15x-3.77x speedup |
| End-to-end causal training | 0.94x-1.16x; some cases regress slightly because planning/execution overhead exceeds saved traffic |
| End-to-end sparse training | 1.00x-1.46x |

Baselines include TransformerEngine and LoongTrain. Benefits grow with more short sequences and sparser masks, precisely where static all-device KV exchange is most wasteful.

## Limitation

- Dense causal batches can see no gain or a small regression.
- Planner complexity grows with block count; smaller blocks expose more optimization but increase overhead.
- The model assumes attention structure is known early enough to plan in the input pipeline.

---

*Reading date: 2026-08*
*Note status: Completed*
