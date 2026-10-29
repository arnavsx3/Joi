# Training data

Joi's default preparation command downloads a small portion of **TinyStories**
by Ronen Eldan and Yuanzhi Li. TinyStories contains synthetic stories generated
using GPT-3.5 and GPT-4. No existing model weights are downloaded or used.

- Dataset: https://huggingface.co/datasets/roneneldan/TinyStories
- Paper: https://arxiv.org/abs/2305.07759
- Dataset revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`
- Dataset license: **Community Data License Agreement — Sharing, Version 1.0**
  (https://cdla.dev/sharing-1-0/)

Raw prefixes and normalized subsets are stored locally in ignored directories.
The generated manifest records the source, license, selection method, hashes,
and split sizes. If you redistribute dataset material or modified subsets,
follow the linked dataset license and retain its notices.

The architecture follows the established decoder-only transformer approach:
causal self-attention, learned token and position embeddings, feed-forward
layers, residual connections, and layer normalization. Joi's implementation
uses PyTorch building blocks; it does not load a pretrained transformer.

- Transformer paper: https://arxiv.org/abs/1706.03762
- GPT-2 architecture reference: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
