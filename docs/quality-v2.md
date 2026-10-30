# Conversation quality experiment, round two

This experiment continues Joi's own 691,712-parameter word model. It uses no
third-party model weights, tokenizer, remote inference, or retrieved replies.
Training data is used during training only. Generated conversations, including
failures, are retained separately from loss measurements.

## Reproduce

From an existing checkout with the original word corpus and chat checkpoint:

```bash
uv run --offline python -m joi.train --data data/processed/dialogue-words --config configs/word-dialogue.json --init-from checkpoints/joi-chat/best.pt --out checkpoints/language-refresh --steps 2000 --batch-size 8 --learning-rate 0.00025 --eval-every 400 --eval-batches 8
uv run --offline python -m joi.tune_companion
uv run --offline python -m joi.tune_companion --init-from checkpoints/companion-v2/best.pt --out checkpoints/companion-v3 --steps 2400 --stop-after 1200 --topic-weight 0.6 --opening-weight 2.5
uv run --offline python -m joi.expand_vocabulary
uv run --offline python -m joi.tune_companion --data data/processed/dialogue-expanded --init-from checkpoints/vocabulary-expanded/initial.pt --out checkpoints/companion-v4 --steps 2000 --topic-weight 0.6 --opening-weight 2.5
uv run --offline python -m joi.chat_evaluate --checkpoint checkpoints/companion-v4/best.pt --data data/processed/dialogue-expanded --probes evaluation/quality-probes-v2.json --out runs/quality-refresh/after.json --with-recall-specialist --tokens 96 --batches 8
```

Each training command requires an empty output directory. The first stage uses
two CPU threads and the original general-dialogue corpus. The second stage mixes
70% project-authored companion batches with 30% general-dialogue replay. Replies
alone receive language-model loss; a context-ranking term additionally rewards
the same answer under its matching conversation over an unrelated conversation.
Its weight is 0.3 and margin is 0.4 nats per target token. The learning rate decays
from 0.0003 to 0.000045 over 2,000 updates. This is a separate experiment with a
fresh optimizer, continuing learned parameters rather than exact optimizer resume.

The new curriculum has 28 everyday situations, multiple replies, informal input,
explicit corrections, requests to listen, and favorite-color/food/drink memory
tasks. The first pass had 1,172 training examples and 300 development examples.
After reviewing its failures, the expanded curriculum has 5,668 training examples
and 320 development examples, including name-copy practice. Situation
user phrasings are disjoint across splits; reference replies and topics are shared.
Memory tasks hold out values while sharing question formats. Some held-out values
already occur in DailyDialog: they are new to this copying task, not new words.
No new downloaded dataset is involved. The authored JSON is versioned in Git;
the corpus, environments and weights remain ignored.

## Evaluation limits

`evaluation/quality-probes-v2.json` was fixed before companion tuning. Its 20
episodes include 38 generated turns, familiar topics phrased differently,
two names, two memory values, and gardening/factual stress tests outside the new
authored situations. Later turns always use the model's own preceding reply.
It is a development suite, not an untouched independent benchmark. The general
model is evaluated with the same 96-token budget, temperature 0.65, top-k 15,
six-token repetition blocking, and seeds before and after training. The old
recall specialist is selected by the same routing rule in both runs.

Checkpoint selection uses the lowest companion reply NLL plus 0.25 times general
reply NLL on fixed development batches. These NLLs average examples equally and
are distinct from the original token-weighted loss. Context preference accuracy
checks whether a reference answer is more likely under its matching prefix than
an unrelated one. It does not assess generated conversation by itself. Literal
memory checks likewise do not establish reliable memory. Read the actual replies.

## First results and refinement

General language continuation completed 2,000 updates in 548.6 seconds. Matched
fixed-window validation loss fell from 1.9287 to 1.8528 nats per learned token,
about 3.9%. Actual conversation remained badly confused. This candidate was not
selected as the chat default.

The first companion pass completed 2,000 updates in 764.5 seconds. Its reply NLL
on the original development examples fell from 2.3263 to 0.6652, while context
preference rose from 70.3% to 92.2%. General reply NLL regressed from 2.1248 to
2.2776. Generated conversations still mixed topics and invented spellings. These
scores are insufficient for deployment. The complete failures are retained in
`evaluation/quality-v2-round-one.json` and `evaluation/quality-v2-language-only.json`.

That pass used the curriculum at commit `e936c2b`. Its training-only copy is
recoverable with `git show e936c2b:data/curated/companion_situations.json`.
The first-pass code is at the same commit. Current commands use the expanded
curriculum, so they are a fresh experiment, not an exact reproduction of that
first pass. Environments and hardware can also change timings and results.

The refinement adds a training-only linear topic head over the last prompt
hidden state. Its classification loss (weight 0.6) makes the network learn the
subject explicitly; the head never selects or retrieves a runtime answer.
The first eight reply tokens receive 2.5 times the language-loss weight. The
inference model's layout and parameter count remain unchanged. Source hashes,
data hashes, head weights, optimizer state, and measured curves are saved locally.

An optional packaging step (`python -m joi.package_companion --checkpoint ...`)
stores a word lexicon derived only from the original and authored training text.
It constrains token combinations during neural generation, allowing new prompt
words such as names. This is a spelling constraint, not an external dictionary,
retrieved reply, or semantic fix. Training text can itself contain errors. It is
evaluated separately from unconstrained neural outputs, and packaging preserves
the neural weights exactly. Temperature/top-k can now also be varied in
`chat_evaluate`; the report records the settings actually used.

`evaluation/quality-followups-v3.json` was authored before examining the refined
model. It adds longer conversations and memory after an intervening topic.
Final results and the deployment decision will be recorded after those checks.
The earlier default checkpoint remains intact.

## Vocabulary growth

The refinement was stopped deliberately after its 1,200-update checkpoint
(509.3 seconds to that checkpoint). A partial interval after that was discarded.
Its development topic accuracy reached 81.25%, but generated replies still
confused topics. This is not reported as a completed 2,400-update run. To repeat
the saved schedule, use `--steps 2400 --stop-after 1200`. This keeps the original
cosine schedule while ending cleanly at the selected checkpoint. Runtime and
source versions remain part of reproducibility; hashes identify the actual run.

Inspection found 186 of 755 lowercased words in the authored training material
fell back to individual characters, including painting, keyboard, and bruv.
The vocabulary-growth stage preserves the original 2,048 token IDs and all
transformer parameters exactly. It appends authored training words first, then
frequent words from the original training corpus, to reach 4,096 pieces.
Validation and probe text never select vocabulary entries. New tied embedding
rows start as the mean of their constituent original Joi embeddings. The model
has 953,856 parameters; its context remains 256 tokens. An automated test checks
the old parameters, token IDs, original-token logits, and tied output weights.

The original corpus is unchanged, with the same text hashes, but its training
token count falls from 2,000,347 to 1,867,571. **Loss values across these two
tokenizers cannot be compared directly.** The vocabulary-expanded continuation
is measured against its own step-zero loss and against actual conversation.

The older recall network still encodes messages using its own 2,048-piece
tokenizer. No token IDs are passed between models. It can therefore continue
serving its existing narrow task after the general model's vocabulary grows.

An optional deterministic beam search explores four neural continuations. It
scores generated sequences by model log probability divided by length to the
power 0.7, preferring completed turns. It has no answer candidates stored in a
database. Beam decoding, lower-temperature sampling, and the spelling constraint
are separate inference experiments; their gains must not be attributed solely
to additional training. Runtime settings are recorded in packaged checkpoints
and can be overridden explicitly by the evaluator.
