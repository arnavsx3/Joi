"""Continue our own network with varied replies and a context-ranking loss."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from joi.checkpoint import load_checkpoint, save_checkpoint
from joi.companion_data import SITUATIONS, records, prepare_examples
from joi.train import load_data
from joi.tune_dialogue import shorten
from joi.tune_recall import batch, make_examples


def per_example_loss(model, sample):
    x, y, weights = sample
    logits, _ = model(x)
    losses = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none").view_as(weights)
    return (losses * weights).sum(1) / weights.sum(1).clamp_min(1)


def collate(examples, context):
    length = max(len(item[0]) for item in examples)
    if length > context:
        raise ValueError("A supervised example exceeds the context")
    x = torch.zeros((len(examples), length), dtype=torch.long)
    y = torch.full_like(x, -100)
    w = torch.zeros_like(x, dtype=torch.float)
    for index, (tokens, labels, weights) in enumerate(examples):
        x[index, :len(tokens)] = torch.tensor(tokens)
        y[index, :len(labels)] = torch.tensor(labels)
        w[index, :len(weights)] = torch.tensor(weights)
    return x, y, w


def contrasting_batch(examples, size, rng, context):
    """Same answer with the right versus an unrelated conversation prefix."""
    if len({topic for topic, _ in examples}) < 2:
        raise ValueError("Context ranking needs at least two different topics")
    positive, negative = [], []
    for _ in range(size):
        topic, sample = rng.choice(examples)
        other_topic, other = rng.choice(examples)
        while other_topic == topic:
            other_topic, other = rng.choice(examples)
        x, y, w = sample
        ox, oy, _ = other
        # Recover the prefix plus the complete target including newline. Labels
        # are shifted once already; the first supervised label predicts target[0].
        start = next(index for index, label in enumerate(y) if label != -100)
        other_start = next(index for index, label in enumerate(oy) if label != -100)
        prefix = ox[:other_start + 1]
        target = y[start:]
        ids = prefix + target
        labels = [-100] * len(prefix) + target
        weights = [0.0] * len(prefix) + [1.0] * len(target)
        wrong = (ids[:-1], labels[1:], weights[1:])
        # Keep the entire answer and nearest prefix for the negative only.
        wrong = tuple(part[-context:] for part in wrong)
        positive.append(sample)
        negative.append(wrong)
    return collate(positive, context), collate(negative, context)


@torch.inference_mode()
def measure(model, examples, replay, context):
    model.eval()
    positive, negative = contrasting_batch(examples, 128, random.Random(918), context)
    # Bound evaluation memory on the laptop.
    def losses(sample):
        return torch.cat([per_example_loss(model, tuple(part[start:start + 16] for part in sample))
                          for start in range(0, sample[0].shape[0], 16)])
    right, wrong = losses(positive), losses(negative)
    general = per_example_loss(model, batch(replay, 64, random.Random(919), context)).mean().item()
    return {"companion_reply_nll": right.mean().item(),
            "context_preference_accuracy": (right < wrong).float().mean().item(),
            "general_reply_nll": general}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-from", type=Path, default=Path("checkpoints/language-refresh/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/dialogue-words"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/companion-v2"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--rank-weight", type=float, default=0.3)
    args = parser.parse_args()
    if min(args.steps, args.batch_size) < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Use positive steps, batch size and learning rate")
    if not math.isfinite(args.rank_weight) or args.rank_weight < 0:
        parser.error("Use a finite nonnegative ranking weight")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Use a new empty output directory")
    torch.set_num_threads(2)
    torch.manual_seed(2031)
    raw = args.init_from.read_bytes()
    model, tokenizer, parent = load_checkpoint(io.BytesIO(raw))
    data_tokenizer, _, hashes, manifest = load_data(args.data)
    if tokenizer.kind != "word" or tokenizer.chars != data_tokenizer.chars:
        parser.error("Use our word checkpoint and matching vocabulary")
    context = model.config.block_size
    train = prepare_examples(tokenizer, context, records("train"))
    val = prepare_examples(tokenizer, context, records("validation"))
    replay = {}
    for split in ("train", "val"):
        conversations = list(dict.fromkeys((args.data / f"{split}.txt").read_text().strip().split("\n\n")))
        replay[split] = shorten(make_examples(tokenizer, conversations), context)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    rng = random.Random(2031)
    args.out.mkdir(parents=True)
    rows, best = [], float("inf")
    started = time.monotonic()
    initialization = {"mode": "own-checkpoint", "source": str(args.init_from),
                      "source_sha256": hashlib.sha256(raw).hexdigest(), "source_step": parent["step"],
                      "source_initialization": parent.get("initialization")}

    def record(step):
        nonlocal best
        scores = measure(model, val, replay["val"], context)
        rows.append({"step": step, **scores, "elapsed_seconds": round(time.monotonic() - started, 3)})
        # A declared development selection rule; generated probes remain separate.
        selection = scores["companion_reply_nll"] + 0.25 * scores["general_reply_nll"]
        state = {"step": step, "optimizer_state": optimizer.state_dict(), "metrics": rows,
                 "data_hashes": hashes, "dataset_manifest": manifest, "initialization": initialization,
                 "curriculum_sha256": hashlib.sha256(SITUATIONS.read_bytes()).hexdigest(),
                 "training_settings": {"seed": 2031, "threads": 2, "batch_size": args.batch_size,
                                       "learning_rate": args.learning_rate, "rank_weight": args.rank_weight},
                 "objective": "70% authored companion batches with assistant-only NLL and context ranking (margin 0.4 nats); 30% DailyDialog/previous persona replay. No runtime retrieval.",
                 "selection": "Lowest companion mean-example reply NLL + 0.25 * general mean-example reply NLL on fixed development batches.",
                 "examples": {"train": len(train), "validation": len(val), "replay": len(replay["train"])},
                 "torch_version": str(torch.__version__)}
        save_checkpoint(args.out / "last.pt", model, tokenizer, **state)
        if selection < best:
            best = selection
            save_checkpoint(args.out / "best.pt", model, tokenizer, **state)
        if step == 0:
            save_checkpoint(args.out / "initial.pt", model, tokenizer, **state)
        # Retain intermediate candidates for transparent generated-output checks.
        save_checkpoint(args.out / f"step-{step}.pt", model, tokenizer, **state)
        (args.out / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(rows[-1]), flush=True)
        model.train()

    record(0)
    for step in range(1, args.steps + 1):
        # Cosine decay gives the short run time to settle.
        lr = args.learning_rate * (0.15 + 0.85 * (1 + math.cos(math.pi * step / args.steps)) / 2)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        if rng.random() < 0.7:
            positive, negative = contrasting_batch(train, args.batch_size, rng, context)
            right = per_example_loss(model, positive)
            wrong = per_example_loss(model, negative)
            loss = right.mean() + args.rank_weight * F.relu(0.4 + right - wrong).mean()
        else:
            loss = per_example_loss(model, batch(replay["train"], args.batch_size, rng, context)).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 400 == 0 or step == args.steps:
            record(step)


if __name__ == "__main__":
    main()
