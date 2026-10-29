"""Prepare an attributed DailyDialog subset plus project-written Joi examples."""

import argparse
import io
import json
from pathlib import Path
import random
import urllib.request
import zipfile

from joi.dialogue import format_conversation
from joi.prepare import digest, distinct_stories
from joi.tokenizer import CharTokenizer

URL = "https://aclanthology.org/attachments/I17-1099.Datasets.zip"
SHA256 = "2de93416a95060057d8b95ef9e2ab9b17a317906f1e352414001a1887f081193"


def read_official_split(raw, split):
    with zipfile.ZipFile(io.BytesIO(raw)) as outer:
        nested = outer.read(f"EMNLP_dataset/{split}.zip")
    with zipfile.ZipFile(io.BytesIO(nested)) as inner:
        text = inner.read(f"{split}/dialogues_{split}.txt").decode("utf-8")
    conversations = []
    for line in text.splitlines():
        turns = [turn.strip() for turn in line.split("__eou__") if turn.strip()]
        turns = turns[:8]
        if len(turns) % 2:
            turns.pop()
        if len(turns) < 2 or any(len(turn) > 240 for turn in turns):
            continue
        messages = [{"role": "user" if index % 2 == 0 else "assistant", "content": turn}
                    for index, turn in enumerate(turns)]
        conversations.append(format_conversation(messages))
    return distinct_stories(conversations)[0]


def curated_conversations(path):
    splits = {"train": [], "val": []}
    for item in json.loads(path.read_text()):
        target = "val" if item.get("holdout") else "train"
        for opener in item["openers"]:
            texts = [opener, item["reply"], item["followup"], item["continuation"]]
            messages = [{"role": "user" if index % 2 == 0 else "assistant", "content": text}
                        for index, text in enumerate(texts)]
            splits[target].append(format_conversation(messages))
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/processed/dialogue"))
    parser.add_argument("--archive", type=Path, default=Path("data/raw/DailyDialog.zip"))
    parser.add_argument("--curated", type=Path, default=Path("data/curated/joi_dialogue_seeds.json"))
    parser.add_argument("--train-dialogues", type=int, default=2000)
    parser.add_argument("--val-dialogues", type=int, default=300)
    parser.add_argument("--persona-repeats", type=int, default=10)
    args = parser.parse_args()
    if min(args.train_dialogues, args.val_dialogues, args.persona_repeats) < 1:
        parser.error("Dialogue counts and persona repeats must be positive")
    if not args.archive.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            raw = response.read(6_000_001)
        if len(raw) > 6_000_000:
            raise ValueError("Dataset archive exceeded the expected size limit")
        if digest(raw) != SHA256:
            raise ValueError("Dataset archive did not match its published pinned hash")
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        args.archive.write_bytes(raw)
    raw = args.archive.read_bytes()
    if digest(raw) != SHA256:
        raise ValueError("Cached dataset archive has an unexpected hash")
    curated = curated_conversations(args.curated)
    train = read_official_split(raw, "train")
    val = read_official_split(raw, "validation")
    random.Random(42).shuffle(train)
    random.Random(43).shuffle(val)
    train = train[:args.train_dialogues] + curated["train"]
    train, train_keys = distinct_stories(train)
    val, _ = distinct_stories(val[:args.val_dialogues] + curated["val"], train_keys)
    # Repetition is a deliberate style weight, not additional unique examples.
    weighted_train = train + curated["train"] * (args.persona_repeats - 1)
    tokenizer = CharTokenizer(["\n"] + [chr(c) for c in range(32, 127)])
    manifest = {
        "dataset": "DailyDialog + project-written Joi dialogues", "format": "joi-dialogue-v1",
        "source_url": URL, "revision": SHA256, "archive_sha256": digest(raw),
        "license": "DailyDialog: CC BY-NC-SA 4.0; curated examples: project-authored",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "curated_sha256": digest(args.curated.read_bytes()),
        "curated_train_conversations": len(curated["train"]),
        "curated_validation_conversations": len(curated["val"]),
        "persona_repeats": args.persona_repeats,
        "selection": "Seeded subsets of official splits; at most eight turns; skip turns over 240 source characters; normalize to ASCII. Whole-conversation exact deduplication. Official test split untouched.",
        "vocab_size": len(tokenizer.chars), "splits": {},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, conversations in (("train", weighted_train), ("val", val)):
        text = "\n".join(conversations)
        tokenizer.encode(text)
        encoded = text.encode()
        (args.output / f"{name}.txt").write_bytes(encoded)
        manifest["splits"][name] = {
            "sha256": digest(encoded), "characters": len(text),
            "unique_conversations": len(set(conversations)), "weighted_conversations": len(conversations),
        }
    (args.output / "tokenizer.json").write_text(json.dumps(tokenizer.chars) + "\n")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
