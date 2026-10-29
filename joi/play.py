"""An offline terminal playground for independent text continuations."""

import argparse
import math
from pathlib import Path

import torch

from joi.checkpoint import load_checkpoint
from joi.generate import generate_text


def run_session(model, tokenizer, *, tokens=200, temperature=0.8, top_k=20,
                seed=42, input_fn=input, output_fn=print):
    output_fn("Joi text playground — type a story starter or another prompt.")
    output_fn("Each prompt starts independently. Use /help for options or /quit to leave.")
    turn = 0
    while True:
        try:
            prompt = input_fn("Prompt> ")
        except (EOFError, KeyboardInterrupt):
            output_fn("\nSession ended.")
            return
        command = prompt.strip()
        if not command:
            continue
        if command == "/quit":
            output_fn("Session ended.")
            return
        if command == "/help":
            output_fn("Enter printable ASCII text. /quit exits. Each successful prompt uses seed + turn; weights stay fixed.")
            continue
        if command.startswith("/"):
            output_fn("Unknown command. Use /help or /quit.")
            continue
        try:
            text = generate_text(model, tokenizer, prompt, tokens=tokens,
                                 temperature=temperature, top_k=top_k, seed=seed + turn)
        except ValueError as exc:
            output_fn(f"Please try again: {exc}")
            continue
        output_fn("Joi> " + text[len(prompt):])
        turn += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/v1/best.pt"))
    parser.add_argument("--tokens", type=int, default=200, help="New characters per prompt")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or args.tokens < 0 or args.top_k < 0:
        parser.error("Use positive threads and nonnegative tokens/top-k")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be finite and positive")
    torch.set_num_threads(args.threads)
    try:
        model, tokenizer, checkpoint = load_checkpoint(args.checkpoint)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    model.eval()
    print(f"Loaded {args.checkpoint} at training step {checkpoint.get('step', 'unknown')}.")
    run_session(model, tokenizer, tokens=args.tokens, temperature=args.temperature,
                top_k=args.top_k, seed=args.seed)


if __name__ == "__main__":
    main()
