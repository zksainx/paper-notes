# Scalable Context Orchestration for Serving LLMs over Voice

<div class="paper-meta" markdown>

**Authors**: Linyi Jiang, Silvery D. Fu, Yifei Zhu  
**Institution**: Shanghai Jiao Tong University; AgenticSys Labs  
**Conference**: SOSP '26 (camera-ready not yet public)  
**Year**: 2026  
**Paper Link**: [Official research page](https://llmovoice.com/research)  

</div>

<div class="paper-tags" markdown>
<span class="paper-tag">voice-llm</span>
<span class="paper-tag">context-management</span>
<span class="paper-tag">streaming</span>
</div>

## Background

A voice session treated as one growing audio stream repeatedly processes history and entangles content, speaking style, and network artifacts. Long sessions become expensive and corrupted network state contaminates future context.

## Methodology

VoicePages store bounded completed interactions with audio, transcript, summary, style, and environment state. VoiceThreads preserve task/topic continuity. A budget-aware Context Projector selects relevant history and representation fidelity; the Context Orchestrator jointly controls content, speaking style, and network/environment behavior.

## Experiment

The public project reports up to 40.1x lower per-turn cost in long conversations while retaining up to 98.7% baseline answer quality, 52.35% lower speaking-rate alignment error, and up to 2.29x better false-trigger rate under network volatility.

## Limitation

- Page/thread construction can lose audio nuance or create retrieval errors.
- Quality and trigger metrics need the final paper's full baseline/setup details.
- This note uses the official research page because the camera-ready PDF is not public yet.

---

*Reading date: 2026-08*
*Note status: Completed from official project; paper pending*
