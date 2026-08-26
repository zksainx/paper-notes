# Murakkab: Resource-Efficient Agentic Workflow Orchestration in Cloud Platforms

<div class="paper-meta" markdown>

**Authors**: Gohar Irfan Chaudhry, Esha Choukse, Haoran Qiu, Inigo Goiri, Rodrigo Fonseca, Adam Belay, Ricardo Bianchini  
**Institution**: MIT CSAIL; Microsoft Azure Research  
**Conference**: OSDI '26  
**Year**: 2026  
**Paper Link**: [USENIX](https://www.usenix.org/conference/osdi26/presentation/chaudhry)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">agentic-workflows</span>
<span class="paper-tag">declarative-runtime</span>
<span class="paper-tag">slo-optimization</span>
</div>

## Background

Agent frameworks expose opaque call sequences while workflow topology, model choice, and hardware provisioning are tuned independently. This prevents end-to-end accuracy/latency/energy/cost optimization.

## Methodology

Murakkab exposes a declarative workflow graph decoupled from deployment. A profile-guided optimizer jointly selects components, models, hardware, and parallelism; an adaptive multi-tenant runtime reconfigures execution as load/SLOs change.

## Experiment

Across diverse workflows, Murakkab reduces GPU use by up to 2.8x, energy by 3.7x, and cost by 4.3x while maintaining target SLOs.

## Limitation

- Requires accuracy profiles for alternative models/workflow configurations.
- Runtime adaptation may oscillate under rapid workload/model changes.
- Declarative graphs must still express dynamic agent control sufficiently.

---

*Reading date: 2026-08*
*Note status: Completed*
