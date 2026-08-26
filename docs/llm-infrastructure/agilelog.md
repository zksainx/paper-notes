# AgileLog: A Forkable Shared Log for Agents on Data Streams

<div class="paper-meta" markdown>

**Authors**: Shreesha G. Bhat, Tony Hong, Michael Noguera, Ramnatthan Alagappan, Aishwarya Ganesan  
**Institution**: University of Illinois Urbana-Champaign  
**Conference**: SOSP '26  
**Year**: 2026  
**Paper Link**: [arXiv:2604.14590](https://arxiv.org/abs/2604.14590)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">shared-log</span>
<span class="paper-tag">agent-isolation</span>
<span class="paper-tag">streaming-data</span>
</div>

## Background

Exploratory agents interfere with real-time stream consumers and can write malformed events. A normal snapshot fork stops receiving parent appends, so it cannot support live analytics or realistic continuous testing.

## Methodology

AgileLog adds severed forks and **continuous forks** that inherit new parent records while isolating child writes; validated cForks can be promoted. Bolt implements them on a diskless log with shared object data, separate brokers, hierarchical indexes, tail-only updates, and lazy tail propagation.

## Experiment

Fork creation is about 50 microseconds independent of log length. Root performance remains unchanged with 100 cForks; metadata scales to 1,000 forks using 8 MB versus 4.4 GB for copying. Kafka experiences 14x/130x mean/P99 latency during agent analysis, while Bolt isolates the root and prevents malformed agent writes from crashing consumers.

## Limitation

- Promote is a restricted single-child merge, not general multi-branch reconciliation.
- Object-store performance/availability becomes part of the log contract.
- Promotable forks can temporarily block consumers and cause throughput dips.

---

*Reading date: 2026-08*
*Note status: Completed*
