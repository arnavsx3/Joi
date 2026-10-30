"""Package our trained checkpoint with a train-only lexical decoding vocabulary."""

import argparse
import hashlib
import json
from pathlib import Path

from joi.checkpoint import load_checkpoint, save_checkpoint
from joi.companion_data import SITUATIONS, records
from joi.lexical import training_words
from joi.train import load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/companion-v2/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/dialogue-words"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/joi-v2/best.pt"))
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new output file; don't overwrite a measured checkpoint")
    model, tokenizer, payload = load_checkpoint(args.checkpoint)
    data_tokenizer, _, hashes, _ = load_data(args.data)
    if tokenizer.kind != "word" or tokenizer.chars != data_tokenizer.chars or hashes != payload["data_hashes"]:
        parser.error("Use the checkpoint's original training corpus and word vocabulary")
    if payload.get("curriculum_sha256") != hashlib.sha256(SITUATIONS.read_bytes()).hexdigest():
        parser.error("Use the exact authored curriculum version recorded by the checkpoint")
    texts = [(args.data / "train.txt").read_text()]
    for _, history, target in records("train"):
        texts.extend(message["content"] for message in history)
        texts.append(target)
    words = training_words("\n".join(texts))
    metadata = {key: value for key, value in payload.items()
                if key not in ("model_state", "model_config", "tokenizer_chars", "tokenizer_type", "format_version")}
    metadata["lexical_words"] = words
    metadata["packaging"] = {
        "source": str(args.checkpoint), "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "neural_weights_changed": False, "lexical_word_count": len(words),
        "lexical_sha256": hashlib.sha256(json.dumps(words).encode()).hexdigest(),
        "note": "Optional word-boundary decoding constraint from training text only, plus words in the current prompt. No response retrieval. This does not fix semantic relevance or make the training lexicon a dictionary.",
    }
    save_checkpoint(args.out, model, tokenizer, **metadata)
    print(json.dumps(metadata["packaging"], indent=2))


if __name__ == "__main__":
    main()
