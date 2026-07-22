# MTP

This category covers Multi-Token Prediction (MTP), a family of training objectives and model augmentations that ask language models to predict more than one future token from a shared prefix. The notes focus on how MTP changes representation learning, planning behavior, and self-speculative inference.

## Research Areas

- Multi-token auxiliary training objectives
- Future-token heads, mask-token predictors, and model-internal drafting
- Connections between MTP, planning, and speculative decoding

## Papers

- [Better & Faster Large Language Models via Multi-token Prediction](better-faster-multi-token-prediction.md) - Introduces a scalable MTP objective with independent future-token heads and shows gains in code generation plus self-speculative decoding.
- [DeepSeek-V3 Technical Report](deepseek-v3.md) - Uses a sequential MTP module as an auxiliary objective in a large MoE model and reports benchmark and decoding benefits.
- [How Transformers Learn to Plan via Multi-Token Prediction](how-transformers-learn-to-plan.md) - Explains why MTP can improve planning through a reverse-reasoning mechanism and cleaner gradients.
- [Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential](your-llm-knows-the-future.md) - Converts pretrained autoregressive models into mask-based future-token predictors with gated LoRA and quadratic verification.
- [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](medusa.md) - Adds trainable future-token decoding heads and tree attention to accelerate inference without a separate draft model.

---

*Continuously updated...*
