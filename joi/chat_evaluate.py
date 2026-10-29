"""Record real multi-turn conversations, plus a separate next-character score."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from joi.chat import ChatEngine
from joi.train import evaluate, load_data


def evaluate_chat(checkpoint, data, probes, *, seed=123, tokens=128, batches=16):
    engine = ChatEngine(checkpoint, tokens=tokens)
    tokenizer, splits, hashes, _ = load_data(data)
    if hashes != engine.checkpoint["data_hashes"] or tokenizer.chars != engine.tokenizer.chars:
        raise ValueError("Use the checkpoint's dialogue dataset and vocabulary")
    losses = evaluate(engine.model, {"val":splits["val"]}, batch_size=8, batches=batches, seed=seed)
    episodes = []
    for index, probe in enumerate(json.loads(probes.read_text())):
        history, results = [], []
        for turn, message in enumerate(probe["turns"]):
            history.append({"role":"user", "content":message})
            turn_seed = seed + index * 100 + turn
            try:
                result = engine.reply(history, seed=turn_seed)
            except ValueError as exc:
                results.append({"user":message, "error":str(exc), "seed":turn_seed})
                break
            history.append({"role":"assistant", "content":result["reply"]})
            results.append({"user":message, **result, "seed":turn_seed})
        expected = probe.get("expected_last_contains")
        last_reply = results[-1].get("reply", "") if results else ""
        episodes.append({**probe, "results":results,
                         "literal_recall_check":expected.lower() in last_reply.lower() if expected else None})
    return {
        "checkpoint":str(checkpoint), "checkpoint_sha256":engine.checkpoint_sha256,
        "phase_step":engine.checkpoint["step"], "model_config":asdict(engine.model.config),
        "initialization":engine.checkpoint.get("initialization"), "dataset_hashes":hashes,
        "tokenizer_type":tokenizer.kind,
        "loss_unit":"nats per character" if tokenizer.kind=="char" else "nats per learned token",
        "validation_loss_nats":losses["val_loss"], "token_perplexity":math.exp(losses["val_loss"]),
        **({"character_perplexity":math.exp(losses["val_loss"])} if tokenizer.kind=="char" else {}),
        "protocol":{"seed":seed, "validation_batches":batches, "validation_batch_size":8,
                    "new_tokens_per_reply":tokens, "temperature":0.65, "top_k":15,
                    "stop":"first generated newline; turn marker is removed"},
        "note":"Development probes, not an independent test set. Each episode uses the model's actual earlier replies. Literal name/topic checks do not establish reliable memory, fluency or cheerfulness.",
        "episodes":episodes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/dialogue-cpu/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/dialogue"))
    parser.add_argument("--probes", type=Path, default=Path("evaluation/chat-probes.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/chat-evaluation.json"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--batches", type=int, default=16)
    args = parser.parse_args()
    if min(args.tokens, args.batches) < 1:
        parser.error("Use positive tokens and batches")
    try:
        report = evaluate_chat(args.checkpoint, args.data, args.probes,
                               seed=args.seed, tokens=args.tokens, batches=args.batches)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validation loss: {report['validation_loss_nats']:.4f} {report['loss_unit']}")
    for episode in report["episodes"]:
        print(f"{episode['id']}: literal recall = {episode['literal_recall_check']}")
        for result in episode["results"]:
            print("  You:",result["user"])
            print("  Joi:",result.get("reply", result.get("error")))
    print(f"Saved actual conversations to {args.out}")


if __name__ == "__main__":
    main()
