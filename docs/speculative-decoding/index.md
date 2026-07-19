# Speculative Decoding

This category covers speculative decoding methods that accelerate autoregressive language models by producing inexpensive draft tokens and verifying them in parallel with the target model while preserving its output distribution.

## Research Areas

- Feature-level and token-level draft modeling
- Dynamic draft trees and parallel block drafting
- Acceptance-aware verification and production scheduling

## Papers

- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](eagle.md) - Predicts target-model features while conditioning on shifted sampled tokens to remove feature uncertainty.
- [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](eagle-2.md) - Builds context-dependent draft trees from calibrated draft-model confidence.
- [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](eagle-3.md) - Replaces constrained feature regression with multi-layer feature fusion and training-time rollout simulation.
- [DFlash: Block Diffusion for Flash Speculative Decoding](dflash.md) - Uses a target-conditioned block diffusion model to draft an entire token block in parallel.
- [Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding](domino.md) - Adds lightweight causal correction to a parallel drafter without repeating the expensive backbone.
- [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](dspark.md) - Combines semi-autoregressive drafting with calibrated, load-aware verification scheduling.

---

*Continuously updated...*
