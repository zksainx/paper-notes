# YoloFS: Information and Control in Agent-Native Filesystems

<div class="paper-meta" markdown>

**Authors**: Shawn Zhong, Junxuan Liao, Jing Liu, Mai Zheng, Andrea C. Arpaci-Dusseau, Remzi H. Arpaci-Dusseau  
**Institution**: University of Wisconsin-Madison; Microsoft Research; Iowa State University  
**Conference**: SOSP '26  
**Year**: 2026  
**Paper Link**: [arXiv:2604.13536](https://arxiv.org/abs/2604.13536)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">agent-filesystem</span>
<span class="paper-tag">undo</span>
<span class="paper-tag">access-control</span>
</div>

## Background

From 290 public incidents, agents delete/corrupt files and leak secrets because neither user nor agent can inspect actual command effects, reverse mutations reliably, or express adaptive access boundaries. Per-command prompts produce approval fatigue without revealing supply-chain side effects.

## Methodology

YoloFS moves policy into the filesystem: staging isolates all writes until commit, continuous snapshots expose/restore history, and progressive permissions gate reads/writes using rules that can evolve during a task. Agents inspect filesystem effects and self-correct; users review sensitive access and final changes.

## Experiment

On 11 hidden-side-effect tasks, agents self-correct 8 and all mutations remain staged. On 112 routine tasks, success matches the baseline with fewer user interactions. I/O matches Ext4 closely, permission overhead is negligible except about 4% for `stat`, and snapshots scale into the hundreds.

## Limitation

- Filesystem mediation cannot control network, process, GPU, or non-file side effects.
- Staged commits still require trustworthy review for subtle semantic damage.
- Snapshot/staging storage grows with large generated artifacts.

---

*Reading date: 2026-08*
*Note status: Completed*
