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


def target_losses(logits, sample):
    _, y, weights = sample
    losses = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none").view_as(weights)
    return (losses * weights).sum(1) / weights.sum(1).clamp_min(1)


def per_example_loss(model, sample):
    logits, _ = model(sample[0])
    return target_losses(logits, sample)


def emphasize_opening(sample, weight):
    x, y, w = sample
    w = w.clone()
    positions = (y != -100).long().cumsum(1)
    w[(y != -100) & (positions <= 8)] *= weight
    return x, y, w


def topic_category(topic):
    return "-".join(topic.split("-")[:2]) if topic.startswith("memory-") else topic


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


def contrasting_batch(examples, size, rng, context, *, return_topics=False):
    """Same answer with the right versus an unrelated conversation prefix."""
    if len({topic for topic, _ in examples}) < 2:
        raise ValueError("Context ranking needs at least two different topics")
    positive, negative, topics = [], [], []
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
        topics.append(topic_category(topic))
    result = collate(positive, context), collate(negative, context)
    return (*result, topics) if return_topics else result


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
    parser.add_argument("--curriculum", type=Path, default=SITUATIONS)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--rank-weight", type=float, default=0.3)
    parser.add_argument("--topic-weight", type=float, default=0.0)
    parser.add_argument("--opening-weight", type=float, default=1.0)
    args = parser.parse_args()
    if min(args.steps, args.batch_size) < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Use positive steps, batch size and learning rate")
    if not math.isfinite(args.rank_weight) or args.rank_weight < 0:
        parser.error("Use a finite nonnegative ranking weight")
    if not math.isfinite(args.topic_weight) or args.topic_weight < 0 or not math.isfinite(args.opening_weight) or args.opening_weight < 1:
        parser.error("Use nonnegative topic weight and opening weight at least one")
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
    train = prepare_examples(tokenizer, context, records("train", args.curriculum))
    val = prepare_examples(tokenizer, context, records("validation", args.curriculum))
    curriculum_sha256 = hashlib.sha256(args.curriculum.read_bytes()).hexdigest()
    categories = sorted({topic_category(topic) for topic, _ in train})
    category_ids = {topic: index for index, topic in enumerate(categories)}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2032)
        topic_head = torch.nn.Linear(model.config.n_embd, len(categories))
    parameters = list(model.parameters()) + list(topic_head.parameters())
    replay = {}
    for split in ("train", "val"):
        conversations = list(dict.fromkeys((args.data / f"{split}.txt").read_text().strip().split("\n\n")))
        replay[split] = shorten(make_examples(tokenizer, conversations), context)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    rng = random.Random(2031)
    args.out.mkdir(parents=True)
    rows, best = [], float("inf")
    started = time.monotonic()
    initialization = {"mode": "own-checkpoint", "source": str(args.init_from),
                      "source_sha256": hashlib.sha256(raw).hexdigest(), "source_step": parent["step"],
                      "source_initialization": parent.get("initialization")}
    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ("tune_companion.py", "companion_data.py", "model.py")}

    def record(step):
        nonlocal best
        scores = measure(model, val, replay["val"], context)
        if args.topic_weight:
            with torch.inference_mode():
                positive, _, topics = contrasting_batch(val, 128, random.Random(920), context, return_topics=True)
                correct = 0
                for start in range(0, 128, 16):
                    x, y, _ = (part[start:start + 16] for part in positive)
                    _, _, hidden = model(x, return_hidden=True)
                    prefix_ends = (y != -100).long().argmax(1)
                    prediction = topic_head(hidden[torch.arange(len(x)), prefix_ends]).argmax(1)
                    expected = torch.tensor([category_ids[topic] for topic in topics[start:start + 16]])
                    correct += (prediction == expected).sum().item()
                scores["topic_accuracy"] = correct / 128
        rows.append({"step": step, **scores, "elapsed_seconds": round(time.monotonic() - started, 3)})
        # A declared development selection rule; generated probes remain separate.
        selection = scores["companion_reply_nll"] + 0.25 * scores["general_reply_nll"]
        state = {"step": step, "optimizer_state": optimizer.state_dict(), "metrics": rows,
                 "data_hashes": hashes, "dataset_manifest": manifest, "initialization": initialization,
                 "curriculum_sha256": curriculum_sha256,
                 "training_source_hashes": source_hashes,
                 "training_settings": {"seed": 2031, "threads": 2, "batch_size": args.batch_size,
                                       "learning_rate": args.learning_rate, "rank_weight": args.rank_weight,
                                       "topic_weight": args.topic_weight, "opening_weight": args.opening_weight},
                 "objective": "70% authored companion batches with assistant-only NLL and context ranking (margin 0.4 nats); 30% DailyDialog/previous persona replay. Optional auxiliary topic classification at the final prompt position and extra weight on the first eight reply tokens. The topic head is training-only. No runtime classification or retrieval.",
                 "auxiliary_topic_categories": categories, "auxiliary_topic_head_state": topic_head.state_dict(),
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
            positive, negative, topics = contrasting_batch(train, args.batch_size, rng, context, return_topics=True)
            positive = emphasize_opening(positive, args.opening_weight)
            negative = emphasize_opening(negative, args.opening_weight)
            logits, _, hidden = model(positive[0], return_hidden=True)
            right = target_losses(logits, positive)
            wrong = per_example_loss(model, negative)
            loss = right.mean() + args.rank_weight * F.relu(0.4 + right - wrong).mean()
            if args.topic_weight:
                prefix_ends = (positive[1] != -100).long().argmax(1)
                topic_logits = topic_head(hidden[torch.arange(args.batch_size), prefix_ends])
                targets = torch.tensor([category_ids[topic] for topic in topics])
                loss = loss + args.topic_weight * F.cross_entropy(topic_logits, targets)
        else:
            loss = per_example_loss(model, batch(replay["train"], args.batch_size, rng, context)).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 400 == 0 or step == args.steps:
            record(step)


if __name__ == "__main__":
    main()
