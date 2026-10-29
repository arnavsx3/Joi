"""Generate text offline using your own trained Joi checkpoint."""

import argparse
from pathlib import Path

import torch

from joi.checkpoint import load_checkpoint


def generate_text(model, tokenizer, prompt, *, tokens=300, temperature=0.8, top_k=20, seed=42):
    """Continue a prompt without updating the model; return prompt plus output."""
    if not prompt:
        raise ValueError("Prompt must contain at least one character")
    encoded = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    torch.manual_seed(seed)
    result = model.generate(encoded, tokens, temperature, top_k)
    return tokenizer.decode(result[0].tolist())


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
        text = generate_text(model, tokenizer, args.prompt, tokens=args.tokens,
                             temperature=args.temperature, top_k=args.top_k, seed=args.seed)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(text)


if __name__ == "__main__":
    main()
