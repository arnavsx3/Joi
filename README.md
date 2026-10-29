# Joi

A small language model, learning to be cheerful.

<img src="assets/joi-portrait.png" alt="Joi, with blue hair in violet and pink neon light" width="280">

The goal is a warm, playful personal chat companion inspired by Joi in
*Blade Runner 2049*. The model stays strictly our own: no third-party pretrained
weights or remote inference services. The portrait is edited from the owner's
supplied image; its [edit prompt](assets/joi-portrait.prompt.txt) is recorded.

## Dialogue training

Joi now has a separate conversation-training path. It starts from our own
random-initialized model's learned weights, expands the context from 128 to 384
characters, and trains on DailyDialog plus original warm, playful Joi examples.
No third-party model weights are used. Context growth preserves learned weights;
only the extra position embeddings start randomly. The optimizer starts fresh.

```bash
uv run python -m joi.prepare_dialogue
uv run --offline python -m joi.train --data data/processed/dialogue --config configs/dialogue.json --init-from checkpoints/v1-longer/best.pt --out checkpoints/dialogue --steps 800 --batch-size 8 --learning-rate 0.0005 --eval-every 200 --eval-batches 4
uv run --offline python -m joi.train --data data/processed/dialogue --config configs/dialogue-cpu.json --init-from checkpoints/dialogue/best.pt --out checkpoints/dialogue-cpu --steps 1600 --batch-size 8 --learning-rate 0.0005 --eval-every 200 --eval-batches 4
```

The first command needs internet on its first run. It verifies the original
DailyDialog archive hash and keeps whole conversations in their official splits.
Project-written scenario families marked `holdout` stay out of training.
The manifest reports unique examples separately from repeated persona examples.
The official test split remains untouched. See
[data sources and licenses](THIRD_PARTY_NOTICES.md).

`--init-from` is different from `--resume`: initialization can change the dataset
and grow the context, while resume restores the same experiment and optimizer.
New runs save `initial.pt` as well as `best.pt` and `last.pt`, allowing a fair
before/after comparison on the dialogue data.

The second stage disables dropout inside attention to enable the faster CPU
attention path; embedding, feed-forward, and residual dropout remain 0.1.
Inference predictions are unchanged when transferring weights between the two
configs because all dropout is disabled during inference. The learned weights
continue, while the optimizer starts fresh for this separate experiment.

## Chat with Joi locally

After preparing and training the dialogue model:

```bash
uv run --offline python -m joi.serve
```

Open **http://127.0.0.1:7860** for the portrait, chat screen, and starter prompts.
The page and neural inference both run on your laptop. Your browser keeps the
current transcript locally; “new conversation” clears it. Recent complete turns
are supplied to the model within its 384-character context, and generation stops
at a turn boundary. This is contextual input, not permanent learned memory.

For a terminal conversation with the same history handling:

```bash
uv run --offline python -m joi.chat
```

Use `/new` to start fresh and `/quit` to leave. This dialogue model is still an
experiment: the interface and history mechanism work, while coherent, cheerful
replies must be checked from actual generated examples.

![Joi chat screen](docs/assets/joi-chat-desktop.jpg)

[Phone layout](docs/assets/joi-chat-mobile.jpg). The default chat checkpoint is
`checkpoints/dialogue-cpu/best.pt`; pass `--checkpoint` to inspect a different
dialogue-trained run.

Record actual multi-turn development probes and a separate validation score:

```bash
uv run --offline python -m joi.chat_evaluate
```

This saves real generated episodes to `runs/chat-evaluation.json`. Later turns
use the model's own earlier replies. Simple literal name/topic checks are
reported separately; they are not a general fluency or memory benchmark. These
development probes are not the untouched official test split.

### Learned word vocabulary experiment

The word experiment trains a **new model from random weights**, with a
2,048-piece vocabulary learned only from its own training text. Common words
are single units; unfamiliar words fall back to ASCII characters, so text still
roundtrips exactly. This is an original simple word/character tokenizer, not a
downloaded tokenizer or pretrained model.

```bash
uv run python -m joi.prepare_dialogue --output data/processed/dialogue-wide --train-dialogues 8000 --persona-repeats 30
uv run --offline python -m joi.prepare_words
uv run --offline python -m joi.train --data data/processed/dialogue-words --config configs/word-dialogue.json --out checkpoints/dialogue-words --steps 4000 --batch-size 8 --learning-rate 0.001 --eval-every 250 --eval-batches 4
uv run --offline python -m joi.serve --checkpoint checkpoints/dialogue-words/best.pt
uv run --offline python -m joi.chat_evaluate --checkpoint checkpoints/dialogue-words/best.pt --data data/processed/dialogue-words --out runs/chat-words.json
```

It uses 691,712 parameters, 256 tokens of context, 8,088 unique training
conversations, and 279 validation conversations after deduplication. Persona
examples are repeated for style weight and counted separately in the manifest.
The vocabulary is fitted on training text only. Word-model loss is measured per
learned token and **cannot be compared numerically with character-model loss**.
Use actual conversation probes to judge the different replies. Checkpoint loading
remains compatible with the earlier character models.

Joi is an educational GPT-style transformer written in Python and PyTorch. It
learns to predict the next character, starting from **random weights**, and runs
locally on a CPU. The first version is a small text generator; fluent dialogue
and a consistently cheerful personality remain goals to evaluate.

## V1

- A character tokenizer with a fixed 96-character ASCII vocabulary.
- A causal transformer: 2 layers, 4 attention heads, 128-dimensional embeddings,
  128-character context, and **425,472 trainable parameters**.
- A bounded subset of TinyStories, with separate training and validation files.
- CPU training, reproducible evaluation, local checkpoints, and training resume.
- Offline text generation and saved sample outputs for human review.
- Tests for causal attention, learning, data separation, checkpoint reload, and
  identical resumed versus uninterrupted training on a small fixture.

**First measured result:** 1,500 CPU updates in about 6 minutes 34 seconds;
training-loop validation loss fell from 4.6275 to 1.3234. Generated samples have
recognizable words and fragments but unreliable grammar and dialogue behavior.
See the [v1 experiment report](docs/v1-results.md) and its learning curve.

**Latest experiment:** continuing the same model to 4,000 updates reduced
matched validation loss by **16.70%**, from 1.3409 to 1.1169 nats per character.
Grammar and conversational replies remain unreliable. The
[longer-training report](docs/longer-training-results.md) contains both sets of
samples, measurements, learning curves, and reproduction commands.

To try the improved local checkpoint:

```bash
uv run --offline python -m joi.play --checkpoint checkpoints/v1-longer/best.pt
```

This requires the continued training run described below. Existing command
defaults still select the original v1 checkpoint.

We implement the architecture using PyTorch's layers, attention operation,
automatic differentiation, and optimizer. We do not download pretrained model
weights. TinyStories itself contains synthetic stories generated by GPT-3.5 and
GPT-4; using those as training examples is distinct from loading their weights.

## Setup on Ubuntu

Requirements: Git, Python, and [uv](https://docs.astral.sh/uv/). This checkout
already has uv available. uv manages the project environment and dependencies;
you do not need a system-wide pip installation or Anaconda.

Run these commands from the Joi repository:

```bash
uv sync --locked
```

`.python-version` selects Python 3.12. uv can download that interpreter if it is
missing. Dependencies are pinned in `uv.lock`. PyTorch comes from its official
**CPU-only** package index; an NVIDIA GPU or CUDA installation is not required.
The initial environment installation needs internet access.

## Prepare data

```bash
uv run python -m joi.prepare
```

This reads just 2 MiB from the official TinyStories training file and 256 KiB
from its validation file at a pinned dataset revision. The raw source files are
much larger; Joi does not download them in full. It keeps complete stories,
normalizes punctuation and whitespace to ASCII, and removes exact duplicate
normalized stories within and across the splits. Similar or paraphrased stories
may remain. Prefix selection is deterministic, not a representative random
sample of the entire dataset.

With the default source revision and download sizes, the prepared subset has:

| Split | Stories | Characters |
| --- | ---: | ---: |
| Training | 2,316 | 2,060,368 |
| Validation | 301 | 257,794 |

Data goes in `data/raw/` and `data/processed/`, both ignored by Git. The prepared
manifest records source URLs, the dataset revision, license, and SHA-256 hashes.
The dataset's license and source are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Re-running preparation reuses cached raw prefixes.

## Train from scratch

```bash
uv run python -m joi.train
```

The default run trains for 1,500 updates with a batch size of 16 and two CPU
threads. It reports training and validation cross-entropy every 100 steps.
**Lower loss is better.** Validation uses the same sampled windows each time,
from text that does not supply training gradients. The untrained step-0 score
provides a baseline.

The first 100 updates took roughly 27 seconds on the development laptop
(i5-11300H, two CPU threads); actual duration depends on other workloads and
hardware. A few minutes of training demonstrates learning, not reliable chat.

Outputs in `checkpoints/v1/`:

- `best.pt`: weights with the lowest measured validation loss.
- `last.pt`: latest saved weights plus optimizer and random-number state.
- `metrics.csv` / `metrics.json`: training and validation loss over time.

Existing output directories are protected against accidental fresh-run
overwrites. For a separate experiment, choose a new directory:

```bash
uv run python -m joi.train --steps 500 --out checkpoints/experiment-2
```

Resume the default run to a **total** of 3,000 updates:

```bash
uv run python -m joi.train --resume checkpoints/v1/last.pt --steps 3000
```

Use the original output directory, dataset, config, seed, batch size, learning
rate, evaluation batch count, and thread count when resuming. These are checked.
Interrupting training leaves the last completed evaluation checkpoint available;
updates since that checkpoint must be repeated. Reproducibility is tested in the
same CPU environment, not guaranteed across PyTorch versions or hardware.

## Generate locally, offline

```bash
uv run --offline python -m joi.generate --prompt "Once upon a time"
```

This reads the best saved weights. It needs neither internet nor an API key
once the environment and checkpoint exist. `--tokens` means **new characters**:

```bash
uv run --offline python -m joi.generate --prompt "A little cat" --tokens 300 --temperature 0.8 --seed 42
```

Lower temperature usually makes output less varied. The model accepts newline
and printable ASCII characters; unsupported prompt characters produce an error.
Only the last 128 characters of context influence each prediction.

To try several prompts without reloading the model:

```bash
uv run --offline python -m joi.play
```

Use `/quit` to leave or `/help` for controls. You can select an experiment with
`--checkpoint checkpoints/v1-longer/best.pt` and adjust `--tokens`,
`--temperature`, or `--seed`. Each prompt starts independently, and using the
playground does not train the model. It is a way to inspect text continuation;
the story-trained model does not retain conversation history.

## Evaluate and test

```bash
uv run python -m joi.evaluate
uv run python -m unittest discover -s tests -v
```

Evaluation saves a held-out next-character score and four reproducible sample
continuations to `runs/v1-evaluation.json`. It uses sampled validation windows,
not the complete validation corpus. Character perplexity is not directly
comparable with word- or subword-token perplexity. The prompts include story
continuation, friendship, comfort, and greeting examples; they are not training
targets. Review samples manually: a lower loss does not prove a cheerful style,
coherent conversation, or factual accuracy.

To draw the saved loss curve, optionally use Matplotlib in a temporary uv
environment (the first run downloads it; it is not required for training):

```bash
uv run --with matplotlib python -m joi.plot
```

The plot is saved to `runs/v1-loss.png`.
For another experiment, pass `--metrics` and `--out`. Use `--min-step 1500` to
zoom in on later training, `--reference-step 1500` to mark the v1 endpoint,
and `--title` to label the experiment.

### Compare experiments fairly

```bash
uv run --offline python -m joi.compare --baseline checkpoints/v1/best.pt --candidate checkpoints/v1-longer/best.pt
```

This evaluates both checkpoints on the same validation windows and prompt
seeds, then writes the scores and both sets of generated samples to
`runs/comparison.json`. Reports include model settings, dataset hashes, the
checkpoint hash, and evaluation settings. Comparisons require matching data,
vocabulary, context length, sampling settings, and PyTorch version. A lower
loss measures improved character prediction; inspect the samples to assess
readability and response quality.

Keep a reference run intact before continuing it as a separate experiment:

```bash
cp -a checkpoints/v1 checkpoints/v1-longer
uv run python -m joi.train --resume checkpoints/v1-longer/last.pt --out checkpoints/v1-longer --steps 4000 --eval-every 250
```

Use a new directory name for each experiment. This changes the training budget
while keeping the model, data, optimizer, and random-number sequence fixed.

## What to learn next

Read [How Joi learns](docs/how-joi-learns.md) for a short explanation of the
training loop and the pieces in this repository. Cloud notebooks such as Colab
are optional for later, larger experiments. This v1 needs neither LM Studio nor
a cloud service.

Next experiments: inspect the generated samples, improve and curate the data,
compare model sizes and training duration, and evaluate a dialogue-specific
dataset before treating Joi as a chatbot.
