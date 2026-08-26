# TStore: Tensor-Centric Compression for Modern AI Model Hubs

<div class="paper-meta" markdown>

**Authors**: Tingfeng Lan, Zirui Wang, Yunjia Zheng, Zhaoyuan Su, Juncheng Yang, Yue Cheng  
**Institution**: University of Virginia; Harvard University  
**Conference**: SOSP '26  
**Year**: 2026  
**Paper Link**: [arXiv:2604.17104](https://arxiv.org/abs/2604.17104)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">model-storage</span>
<span class="paper-tag">delta-compression</span>
<span class="paper-tag">tensor-indexing</span>
</div>

## Background

Hugging Face grew beyond 76 PB by end-2025; 99% is fine-tuned variants, but only 25.8% of 2.7M models expose usable lineage. A single declared base is suboptimal because each tensor drifts differently and may be closest to tensors in another variant.

## Methodology

TStore decomposes models into tensors. TensorSketch fingerprints bit-level distribution/layout; a regressor predicts pair compressibility; FlexSplit incrementally builds multi-center clusters and selects per-tensor bases; parallel codecs encode/decode chosen deltas.

## Experiment

On 2,890 Hugging Face models, storage falls 70.5% (3.39x), 37% below the prior design. Ingestion/decompression reach 22.9/28.4 GB/s, 3.86x/1.49x the next best. TensorSketch reaches 25K QPS and 20,082x speedup over exact bit distance with perfect Recall@1 in reported families.

## Limitation

- Evaluation samples 40.11 TB, far below a full multi-PB hub.
- Cross-model tensor bases complicate deletion, garbage collection, and corruption domains.
- Lossless delta savings vary sharply by model family and training history.

---

*Reading date: 2026-08*
*Note status: Completed*
