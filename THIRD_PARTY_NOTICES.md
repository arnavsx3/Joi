# Training data

## Dialogue experiment

The dialogue preparation command uses **DailyDialog: A Manually Labelled
Multi-turn Dialogue Dataset** (Li et al., 2017).

- Paper and original dataset attachment: https://aclanthology.org/I17-1099/
- Archive: https://aclanthology.org/attachments/I17-1099.Datasets.zip
- Archive SHA-256: `2de93416a95060057d8b95ef9e2ab9b17a317906f1e352414001a1887f081193`
- Author-associated dataset card and license declaration:
  https://huggingface.co/datasets/li2017dailydialog/daily_dialog
- License: **Creative Commons Attribution-NonCommercial-ShareAlike 4.0**
  (https://creativecommons.org/licenses/by-nc-sa/4.0/).

Joi selects bounded subsets of the official train and validation splits, limits
turn length, normalizes punctuation and whitespace to ASCII, labels speakers
`User` and `Joi`, and removes exact duplicate complete conversations. The test
split is not used for training or validation. Downloaded and adapted dataset
files remain in Git-ignored directories. Dataset redistribution must follow the
linked license, including attribution, noncommercial use, and share-alike terms.

`data/curated/joi_dialogue_seeds.json` contains original project-written dialogue
examples, with reserved holdout scenario families. Training repeats the curated
examples to increase their style weight; repeats are not extra unique examples.

## Story experiments

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
