"""Generate text offline using your own trained Joi checkpoint."""

import argparse
from pathlib import Path

import torch

from joi.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/v1/best.pt"))
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=300, help="Number of new characters")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20, help="0 disables top-k filtering")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if not args.prompt:
        parser.error("--prompt must contain at least one character")
    torch.set_num_threads(args.threads)
    try:
        model, tokenizer, _ = load_checkpoint(args.checkpoint)
        tokens = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
        torch.manual_seed(args.seed)
        result = model.generate(tokens, args.tokens, args.temperature, args.top_k)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(tokenizer.decode(result[0].tolist()))


if __name__ == "__main__":
    main()
