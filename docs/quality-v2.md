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
uv run --offline python -m joi.chat_evaluate --checkpoint checkpoints/companion-v2/best.pt --data data/processed/dialogue-words --probes evaluation/quality-probes-v2.json --out runs/quality-refresh/after.json --with-recall-specialist --tokens 96 --batches 8
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
tasks. There are 1,172 training examples and 300 development examples. Situation
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

Results and the deployment decision will be recorded after training and generated
conversation checks complete. The earlier default checkpoint remains intact.
