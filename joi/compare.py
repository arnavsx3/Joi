"""Compare two local checkpoints using identical data, windows and prompts."""

import argparse
import json
import math
from pathlib import Path

import torch

from joi.evaluate import evaluate_checkpoint


def compare_reports(baseline, candidate):
    for key in ("dataset", "vocabulary_sha256", "evaluation", "generation", "torch_version"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Cannot compare reports with different {key}")
    prompt_keys = lambda report: [(item["prompt"], item["seed"]) for item in report["samples"]]
    if prompt_keys(baseline) != prompt_keys(candidate):
        raise ValueError("Cannot compare reports with different prompts or sample seeds")
    before, after = baseline["validation_loss_nats"], candidate["validation_loss_nats"]
    if not all(math.isfinite(value) and value > 0 for value in (before, after)):
        raise ValueError("Comparison needs finite, positive validation losses")
    return {
        "schema_version": 1,
        "loss_change_nats": after - before,
        "relative_loss_reduction_percent": 100 * (before - after) / before,
        "lower_loss_checkpoint": "candidate" if after < before else "baseline" if before < after else "tie",
        "interpretation": "Negative loss change means improved next-character prediction on identical sampled validation windows. This is not a dialogue or cheerfulness score.",
        "baseline": baseline, "candidate": candidate,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--prompts", type=Path, default=Path("evaluation/prompts.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/comparison.json"))
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        reports = [
            evaluate_checkpoint(path, args.data, args.prompts, batches=args.batches,
                                tokens=args.tokens, seed=args.seed,
                                temperature=args.temperature, top_k=args.top_k)
            for path in (args.baseline, args.candidate)
        ]
        comparison = compare_reports(*reports)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"Baseline: {reports[0]['validation_loss_nats']:.4f} nats/character")
    print(f"Candidate: {reports[1]['validation_loss_nats']:.4f} nats/character")
    print(f"Relative loss reduction: {comparison['relative_loss_reduction_percent']:.2f}%")
    print(f"Saved matched samples and provenance to {args.out}")


if __name__ == "__main__":
    main()
