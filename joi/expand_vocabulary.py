"""Grow our own word vocabulary while preserving all previously learned rows."""

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from joi.checkpoint import load_checkpoint, save_checkpoint
from joi.companion_data import SITUATIONS, records
from joi.model import Joi
from joi.tokenizer import PIECES, WordTokenizer
from joi.train import load_data


def expand_model(source, old_tokenizer, tokenizer):
    if tokenizer.chars[:len(old_tokenizer.chars)] != old_tokenizer.chars:
        raise ValueError("Vocabulary growth must preserve every existing token ID")
    model = Joi(replace(source.config, vocab_size=len(tokenizer.chars)))
    state = model.state_dict()
    for key, value in source.state_dict().items():
        if key not in ("token_embedding.weight", "head.weight"):
            state[key].copy_(value)
    embedding = state["token_embedding.weight"]
    with torch.no_grad():
        embedding[:len(old_tokenizer.chars)].copy_(source.token_embedding.weight)
        for index, piece in enumerate(tokenizer.chars[len(old_tokenizer.chars):], len(old_tokenizer.chars)):
            embedding[index].copy_(source.token_embedding.weight[old_tokenizer.encode(piece)].mean(0))
        state["head.weight"].copy_(embedding)
    model.load_state_dict(state)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/companion-v3/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/dialogue-words"))
    parser.add_argument("--out-data", type=Path, default=Path("data/processed/dialogue-expanded"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/vocabulary-expanded/initial.pt"))
    parser.add_argument("--vocab-size", type=int, default=4096)
    args = parser.parse_args()
    if args.out.exists() or args.out_data.exists():
        parser.error("Use new checkpoint and dataset paths")
    torch.set_num_threads(2)
    torch.manual_seed(2033)
    source, old_tokenizer, parent = load_checkpoint(args.checkpoint)
    data_tokenizer, _, hashes, manifest = load_data(args.data)
    if old_tokenizer.kind != "word" or data_tokenizer.chars != old_tokenizer.chars or hashes != parent["data_hashes"]:
        parser.error("Use our matching word checkpoint and corpus")
    # No validation text or probe text is used to select vocabulary entries.
    training_text = (args.data / "train.txt").read_text()
    authored_text = "\n".join("\n".join(m["content"] for m in history) + "\n" + target
                              for _, history, target in records("train"))
    required = sorted({piece for piece in PIECES.findall(authored_text)
                       if len(piece) > 1 and piece not in old_tokenizer.stoi})
    pieces = old_tokenizer.chars + required
    if len(pieces) > args.vocab_size:
        parser.error("Vocabulary size must fit the parent and authored training words")
    seen = set(pieces)
    counts = Counter(piece for piece in PIECES.findall(training_text) if len(piece) > 1)
    for piece in sorted(counts, key=lambda word: (-counts[word], word)):
        if len(pieces) >= args.vocab_size:
            break
        if piece not in seen:
            pieces.append(piece)
            seen.add(piece)
    tokenizer = WordTokenizer(pieces)
    model = expand_model(source, old_tokenizer, tokenizer)
    manifest = {**manifest, "vocab_size": len(pieces), "splits": {},
                "tokenizer": {"type": "word", "fit": "Original training text plus authored training examples only; no validation/probes.",
                              "parent_vocabulary_size": len(old_tokenizer.chars), "authored_additions": len(required),
                              "curriculum_sha256": hashlib.sha256(SITUATIONS.read_bytes()).hexdigest()}}
    args.out_data.mkdir(parents=True)
    for split in ("train", "val"):
        raw = (args.data / f"{split}.txt").read_bytes()
        (args.out_data / f"{split}.txt").write_bytes(raw)
        previous = parent["dataset_manifest"]["splits"][split]
        manifest["splits"][split] = {**previous, "tokens": len(tokenizer.encode(raw.decode()))}
    (args.out_data / "tokenizer.json").write_text(json.dumps({"type": "word", "pieces": pieces}) + "\n")
    (args.out_data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    initialization = {"mode": "own-checkpoint-vocabulary-growth", "source": str(args.checkpoint),
                      "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      "source_step": parent["step"], "source_initialization": parent.get("initialization"),
                      "old_vocabulary_size": len(old_tokenizer.chars), "new_vocabulary_size": len(pieces),
                      "note": "All old embeddings, transformer parameters and positions copied exactly. New tied embedding rows are means of their original Joi token embeddings. No third-party weights."}
    save_checkpoint(args.out, model, tokenizer, step=0, data_hashes=hashes,
                    dataset_manifest=manifest, initialization=initialization)
    print(json.dumps({"initialization": initialization, "parameters": sum(p.numel() for p in model.parameters()),
                      "splits": manifest["splits"]}, indent=2))


if __name__ == "__main__":
    main()
