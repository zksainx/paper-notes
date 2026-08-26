# UEP: Portable Expert-Parallel Communication

<div class="paper-meta" markdown>

**Authors**: Ziming Mao, Yihan Zhang, Chihan Cui, Zhen Huang, Kaichao You, et al.  
**Institution**: UC Berkeley; UC Davis; UW-Madison; AMD; AWS; Tsinghua; Broadcom  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/mao-ziming-uep)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">expert-parallelism</span>
<span class="paper-tag">rdma</span>
<span class="paper-tag">portability</span>
</div>

## Background

DeepEP-style GPU-initiated RDMA tightly couples each GPU and NIC vendor, creating O(m x n) integration work and excluding EFA-like NICs with different ordering semantics.

## Methodology

UEP sends compact routing commands through a high-throughput GPU-CPU channel; multithreaded CPU proxies issue GPUDirect RDMA. RDMA immediate data emulates dispatch/combine ordering, reducing hardware support to O(m) GPU-side integration plus portable `libibverbs` NIC support.

## Experiment

On EFA, dispatch/combine throughput improves 2.1x; SGLang token throughput rises up to 40% on NVIDIA+EFA; DeepSeek-V3 training improves up to 45% on 16 AMD+Broadcom nodes.

## Limitation

- CPU proxies reintroduce host scheduling and NUMA concerns.
- Very fine token traffic can saturate the control channel.
- Ordering emulation complexity differs across NICs.

---

*Reading date: 2026-08*
*Note status: Completed*
