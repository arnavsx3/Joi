# Joi

A small language model, learning to be cheerful.

Joi is a learning project: build a small GPT-style transformer in Python and
PyTorch, train it from randomly initialized weights, and run it locally on a
laptop. The goal is to understand how a language model learns, one experiment
at a time.

## The idea

Start with a tiny character-level model that learns to predict the next
character in a piece of text. Train it on a curated dataset with a warm,
playful tone, then see what it can generate.

Joi follows a published transformer architecture. We will write the model and
training code ourselves using PyTorch's building blocks and train our own
weights.

## Planned stack

- **Python** for data preparation, training, and text generation.
- **PyTorch (CPU)** for the neural network, automatic differentiation, and
  optimization.
- **A character-level tokenizer** to turn text into numbers and back.
- **Local checkpoints** to save and reload learned weights.

## First milestones

- [ ] Set up a reproducible Python environment.
- [ ] Prepare training and validation data.
- [ ] Implement a small causal transformer.
- [ ] Train a baseline and track training and validation loss.
- [ ] Save a checkpoint and generate text from it offline.
- [ ] Evaluate cheerful responses on examples held out from training.

## Current status

Project initialized. Model code, a dataset, and trained weights are not yet
included. Setup and usage instructions will be added with the implementation.

The first version is a small educational text generator. Coherent conversation
and a consistent cheerful style are goals to evaluate, rather than capabilities
it already has.
