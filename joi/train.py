"""Train Joi from random weights, or resume a local training checkpoint."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import time

import torch

from joi.checkpoint import load_checkpoint, save_checkpoint
from joi.model import Joi, ModelConfig
from joi.tokenizer import tokenizer_from_state


def initialize_from_checkpoint(path, config, tokenizer):
    raw = path.read_bytes()
    source, source_tokenizer, payload = load_checkpoint(io.BytesIO(raw))
    previous = asdict(source.config)
    requested = asdict(config)
    if source_tokenizer.chars != tokenizer.chars:
        raise ValueError("Initialization requires the same vocabulary")
    if any(previous[key] != requested[key] for key in previous if key not in ("block_size", "attention_dropout")):
        raise ValueError("Initialization can change context length and attention dropout, but not the parameter layout")
    if config.block_size < source.config.block_size:
        raise ValueError("Initialization cannot shrink the original context")
    model = Joi(config)
    state = model.state_dict()
    for key, value in source.state_dict().items():
        if key == "position_embedding.weight":
            state[key][:source.config.block_size].copy_(value)
        else:
            state[key].copy_(value)
    model.load_state_dict(state)
    return model, {
        "mode": "own-checkpoint", "source": str(path),
        "source_sha256": hashlib.sha256(raw).hexdigest(), "source_step": payload.get("step"),
        "source_model_config": previous,
        "source_initialization": payload.get("initialization", {"mode":"random"}),
        "new_positions": config.block_size - source.config.block_size,
        "note": "Copied learned parameters; added position embeddings keep their random initialization. New optimizer and step counter.",
    }


def sample_batch(data, block_size, batch_size, generator):
    if len(data) <= block_size:
        raise ValueError(f"Each data split needs more than {block_size} characters")
    starts = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    chunks = data[starts[:, None] + torch.arange(block_size + 1)]
    return chunks[:, :-1], chunks[:, 1:]


@torch.inference_mode()
def evaluate(model, splits, batch_size, batches, seed):
    was_training = model.training
    model.eval()
    result = {}
    try:
        # Reset a separate generator: every evaluation uses the same examples and
        # never changes the training sampler or dropout random-number streams.
        for offset, (name, data) in enumerate(splits.items()):
            rng = torch.Generator().manual_seed(seed + 1000 + offset)
            losses = []
            for _ in range(batches):
                x, y = sample_batch(data, model.config.block_size, batch_size, rng)
                _, loss = model(x, y)
                losses.append(loss.item())
            result[f"{name}_loss"] = sum(losses) / len(losses)
    finally:
        model.train(was_training)
    return result


def load_data(directory):
    tokenizer = tokenizer_from_state(json.loads((directory / "tokenizer.json").read_text()))
    manifest = json.loads((directory / "manifest.json").read_text())
    splits = {}
    hashes = {}
    for name in ("train", "val"):
        raw = (directory / f"{name}.txt").read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != manifest["splits"][name]["sha256"]:
            raise ValueError(f"{name}.txt does not match its manifest; prepare data again")
        hashes[name] = sha
        splits[name] = torch.tensor(tokenizer.encode(raw.decode("utf-8")), dtype=torch.long)
    return tokenizer, splits, hashes, manifest


def write_metrics(directory, rows):
    temporary = directory / "metrics.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(directory / "metrics.csv")
    temporary = directory / "metrics.json.tmp"
    temporary.write_text(json.dumps(rows, indent=2) + "\n")
    temporary.replace(directory / "metrics.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--config", type=Path, default=Path("configs/tiny.json"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/v1"))
    initialization_args = parser.add_mutually_exclusive_group()
    initialization_args.add_argument("--resume", type=Path)
    initialization_args.add_argument("--init-from", type=Path, help="Continue our own learned weights on a new dataset, with a fresh optimizer")
    parser.add_argument("--steps", type=int, default=1500, help="Total target step, including previous training")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("steps", "batch_size", "eval_every", "eval_batches", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not args.resume and args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory is not empty; use --resume or a new --out directory")
    if args.resume and args.out.resolve() != args.resume.resolve().parent:
        parser.error("Resume in the original output directory so the best checkpoint is preserved")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    try:
        tokenizer, splits, hashes, manifest = load_data(args.data)
        config = ModelConfig(vocab_size=len(tokenizer.chars), **json.loads(args.config.read_text()))
        for data in splits.values():
            if len(data) <= config.block_size:
                raise ValueError("Prepared splits are shorter than the model context")
        training_settings = {
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "eval_batches": args.eval_batches, "threads": args.threads, "seed": args.seed,
        }
        if args.resume:
            model, saved_tokenizer, checkpoint = load_checkpoint(args.resume)
            if hashes != checkpoint["data_hashes"] or tokenizer.chars != saved_tokenizer.chars:
                raise ValueError("Resume requires the same dataset and vocabulary")
            if asdict(config) != asdict(ModelConfig(**checkpoint["model_config"])):
                raise ValueError("Resume requires the original model config")
            if training_settings != checkpoint["training_settings"]:
                raise ValueError("Resume requires the original batch size, learning rate, evaluation batches, threads and seed")
            first_step = checkpoint["step"]
            initialization = checkpoint.get("initialization", {"mode": "random"})
            if args.steps <= first_step:
                raise ValueError(f"--steps must exceed the saved step ({first_step})")
        else:
            if args.init_from:
                model, initialization = initialize_from_checkpoint(args.init_from, config, tokenizer)
            else:
                model, initialization = Joi(config), {"mode": "random"}
            checkpoint = {}
            first_step = 0
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    batch_rng = torch.Generator().manual_seed(args.seed + 1)
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        batch_rng.set_state(checkpoint["batch_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
    rows = checkpoint.get("metrics", [])
    best_loss = checkpoint.get("best_val_loss", float("inf"))
    best_step = checkpoint.get("best_step", 0)
    elapsed_before = checkpoint.get("elapsed_seconds", 0.0)
    args.out.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Joi: {parameter_count:,} parameters | CPU | {args.threads} threads | context {config.block_size}", flush=True)
    print(f"Data: {len(splits['train']):,} training / {len(splits['val']):,} validation {tokenizer.kind} tokens", flush=True)
    print("Resuming learned weights." if args.resume else
          f"Initializing from our own checkpoint: {args.init_from}" if args.init_from else
          "Starting from randomly initialized weights.", flush=True)
    started = time.monotonic()

    def record(step):
        nonlocal best_loss, best_step
        losses = evaluate(model, splits, args.batch_size, args.eval_batches, args.seed)
        if not all(math.isfinite(value) for value in losses.values()):
            raise RuntimeError("Non-finite evaluation loss; training stopped")
        elapsed = elapsed_before + time.monotonic() - started
        row = {"step": step, **losses, "elapsed_seconds": round(elapsed, 3)}
        rows.append(row)
        improved = losses["val_loss"] < best_loss
        if improved:
            best_loss, best_step = losses["val_loss"], step
        state = {
            "step": step, "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(), "batch_rng_state": batch_rng.get_state(),
            "data_hashes": hashes, "dataset_manifest": manifest,
            "training_settings": training_settings, "metrics": rows,
            "best_val_loss": best_loss, "best_step": best_step,
            "elapsed_seconds": elapsed, "parameter_count": parameter_count,
            "torch_version": str(torch.__version__),
            "initialization": initialization,
        }
        save_checkpoint(args.out / "last.pt", model, tokenizer, **state)
        if improved:
            save_checkpoint(args.out / "best.pt", model, tokenizer, **state)
        if step == 0:
            save_checkpoint(args.out / "initial.pt", model, tokenizer, **state)
        write_metrics(args.out, rows)
        print(f"step {step:5d} | train {losses['train_loss']:.4f} | val {losses['val_loss']:.4f} | {elapsed:.1f}s", flush=True)

    model.train()
    if not args.resume:
        record(0)
    try:
        for step in range(first_step + 1, args.steps + 1):
            x, y = sample_batch(splits["train"], config.block_size, args.batch_size, batch_rng)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; use the last saved checkpoint")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step % args.eval_every == 0 or step == args.steps:
                record(step)
    except KeyboardInterrupt:
        print("\nInterrupted. Resume from last.pt at the last completed evaluation.", flush=True)
        return
    print(f"Finished. Best validation loss: {best_loss:.4f} at step {best_step}.", flush=True)
    print(f"Generate: uv run python -m joi.generate --checkpoint {args.out / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
