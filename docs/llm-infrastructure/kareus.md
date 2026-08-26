# Kareus: Joint Reduction of Dynamic and Static Energy in Large Model Training

<div class="paper-meta" markdown>

**Authors**: Ruofan Wu et al.  
**Institution**: University of Michigan  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/wu-ruofan)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">energy</span>
<span class="paper-tag">frequency-scaling</span>
<span class="paper-tag">kernel-scheduling</span>
</div>

## Background

Frequency scaling reduces dynamic energy but changes compute/communication overlap; fine-grained kernel scheduling reduces idle/static energy but its best SM allocation depends on frequency. Optimizing them independently misses schedules whose time/energy differs by up to 3.29x for identical work.

## Methodology

Kareus partitions the joint search over GPU frequency, communication-kernel SM allocation, and launch timing. A multi-pass multi-objective optimizer solves local subproblems and constructs a Pareto frontier for time versus energy.

## Experiment

Compared with prior systems, Kareus saves up to 28.3% energy at equal training time or shortens training by up to 27.5% at equal energy.

## Limitation

- Requires stable kernel profiles and hardware frequency controls.
- Local decomposition may miss global interactions across partitions/nodes.
- A Pareto frontier still requires an operator policy for choosing time versus energy.

---

*Reading date: 2026-08*
*Note status: Completed*
