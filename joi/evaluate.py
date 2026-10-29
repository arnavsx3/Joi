"""Measure held-out next-character loss and save reproducible sample outputs."""

import argparse
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path

import torch

from joi.checkpoint import load_checkpoint
from joi.train import evaluate, load_data


def evaluate_checkpoint(checkpoint_path, data_directory, prompts_path, *, batches=32,
                        tokens=200, seed=123, temperature=0.8, top_k=20):
    """Use a fixed protocol so two saved models can be compared fairly."""
    if batches < 1 or tokens < 0:
        raise ValueError("Use positive batches and a nonnegative token count")
    if not math.isfinite(temperature) or temperature <= 0 or top_k < 0:
        raise ValueError("Use a finite, positive temperature and nonnegative top-k")
    # Read one snapshot: a concurrent training save can replace this path while
    # evaluation runs. The reported hash must identify the bytes we loaded.
    checkpoint_bytes = checkpoint_path.read_bytes()
    model, tokenizer, checkpoint = load_checkpoint(io.BytesIO(checkpoint_bytes))
    data_tokenizer, splits, hashes, manifest = load_data(data_directory)
    if hashes != checkpoint["data_hashes"] or data_tokenizer.chars != tokenizer.chars:
        raise ValueError("Evaluation data must match the checkpoint's recorded dataset")
    losses = evaluate(model, {"val": splits["val"]}, batch_size=16, batches=batches, seed=seed)
    samples = []
    for index, item in enumerate(json.loads(prompts_path.read_text())):
        prompt = item["prompt"]
        if not prompt:
            raise ValueError("Evaluation prompts must be nonempty")
        torch.manual_seed(seed + index)
        x = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        result = model.generate(x, max_new_tokens=tokens, temperature=temperature, top_k=top_k)
        samples.append({**item, "seed": seed + index,
                        "continuation": tokenizer.decode(result[0, len(prompt):].tolist())})
    return {
        "schema_version": 2,
        "checkpoint": str(checkpoint_path), "step": checkpoint["step"],
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "model_config": asdict(model.config),
        "vocabulary_sha256": hashlib.sha256(json.dumps(tokenizer.chars).encode()).hexdigest(),
        "dataset": {"source": manifest.get("dataset"), "revision": manifest.get("revision"),
                    "split_sha256": hashes},
        "torch_version": str(torch.__version__),
        "validation_loss_nats": losses["val_loss"],
        "character_perplexity": math.exp(losses["val_loss"]),
        "evaluation": {"batches": batches, "batch_size": 16,
                       "context": model.config.block_size, "seed": seed,
                       "note": "Sampled windows from the held-out split; not the full validation corpus."},
        "generation": {"temperature": temperature, "top_k": top_k, "new_characters": tokens},
        "limitations": "Loss measures next-character prediction. Samples require human review; this does not establish cheerful or conversational ability.",
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/v1/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--prompts", type=Path, default=Path("evaluation/prompts.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/v1-evaluation.json"))
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.batches, args.threads) < 1 or args.tokens < 0:
        parser.error("Use positive batches/threads and a nonnegative token count")
    torch.set_num_threads(args.threads)
    try:
        report = evaluate_checkpoint(
            args.checkpoint, args.data, args.prompts, batches=args.batches,
            tokens=args.tokens, seed=args.seed, temperature=args.temperature, top_k=args.top_k,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
