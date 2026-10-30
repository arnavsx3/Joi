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
uv run --offline python -m joi.tune_companion --init-from checkpoints/companion-v2/best.pt --out checkpoints/companion-v3 --steps 2400 --topic-weight 0.6 --opening-weight 2.5
uv run --offline python -m joi.chat_evaluate --checkpoint checkpoints/companion-v3/best.pt --data data/processed/dialogue-words --probes evaluation/quality-probes-v2.json --out runs/quality-refresh/after.json --with-recall-specialist --tokens 96 --batches 8
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
