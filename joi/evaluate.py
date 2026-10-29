"""Measure held-out next-character loss and save reproducible sample outputs."""

import argparse
import json
import math
from pathlib import Path

import torch

from joi.checkpoint import load_checkpoint
from joi.train import evaluate, load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/v1/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--prompts", type=Path, default=Path("evaluation/prompts.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/v1-evaluation.json"))
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.batches, args.threads) < 1 or args.tokens < 0:
        parser.error("Use positive batches/threads and a nonnegative token count")
    torch.set_num_threads(args.threads)
    try:
        model, tokenizer, checkpoint = load_checkpoint(args.checkpoint)
        data_tokenizer, splits, hashes, _ = load_data(args.data)
        if hashes != checkpoint["data_hashes"] or data_tokenizer.chars != tokenizer.chars:
            raise ValueError("Evaluation data must match the checkpoint's recorded dataset")
        losses = evaluate(model, {"val": splits["val"]}, batch_size=16, batches=args.batches, seed=args.seed)
        samples = []
        for index, item in enumerate(json.loads(args.prompts.read_text())):
            prompt = item["prompt"]
            if not prompt:
                raise ValueError("Evaluation prompts must be nonempty")
            torch.manual_seed(args.seed + index)
            x = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
            result = model.generate(x, max_new_tokens=args.tokens, temperature=0.8, top_k=20)
            samples.append({**item, "seed": args.seed + index,
                            "continuation": tokenizer.decode(result[0, len(prompt):].tolist())})
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    report = {
        "checkpoint": str(args.checkpoint), "step": checkpoint["step"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "validation_loss_nats": losses["val_loss"],
        "character_perplexity": math.exp(losses["val_loss"]),
        "evaluation": {"batches": args.batches, "batch_size": 16,
                       "context": model.config.block_size, "seed": args.seed,
                       "note": "Sampled windows from the held-out split; not the full validation corpus."},
        "generation": {"temperature": 0.8, "top_k": 20, "new_characters": args.tokens},
        "limitations": "Loss measures next-character prediction. Samples require human review; this does not establish cheerful or conversational ability.",
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
