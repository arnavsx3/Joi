"""Download bounded TinyStories prefixes and prepare disjoint story splits."""

import argparse
import hashlib
import json
from pathlib import Path
import unicodedata
import urllib.request

from joi.tokenizer import CharTokenizer


DATASET = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
SOURCE = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"
DELIMITER = "<|endoftext|>"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalize(text):
    replacements = str.maketrans({
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " ",
    })
    text = unicodedata.normalize("NFKD", text.translate(replacements))
    text = text.encode("ascii", errors="ignore").decode("ascii")
    return "\n".join(" ".join(line.split()) for line in text.splitlines()).strip()


def stories_from_prefix(raw):
    # Discard the unfinished final story and any partial final UTF-8 character.
    parts = raw.decode("utf-8", errors="ignore").split(DELIMITER)
    return [story for part in parts[:-1] if (story := normalize(part))]


def distinct_stories(stories, excluded=None):
    seen = set(excluded or ())
    result = []
    for story in stories:
        key = digest(story.encode("utf-8"))
        if key not in seen:
            result.append(story)
            seen.add(key)
    return result, seen


def download_prefix(filename, limit, raw_dir):
    path = raw_dir / f"{filename}.{limit}.prefix"
    if path.exists():
        raw = path.read_bytes()
        if len(raw) != limit:
            raise ValueError(f"Incomplete cached download: remove {path} and retry")
        return raw
    print(f"Downloading the first {limit:,} bytes of {filename}...", flush=True)
    request = urllib.request.Request(
        f"{SOURCE}/{filename}", headers={"User-Agent": "Joi/0.1 educational project"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read(limit)
    if len(raw) != limit:
        raise ValueError(f"Expected {limit:,} bytes, received {len(raw):,}")
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--val-bytes", type=int, default=256 * 1024)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    if min(args.train_bytes, args.val_bytes) < 4096:
        parser.error("Use at least 4096 bytes per split to include complete stories")
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    train_raw = download_prefix("TinyStories-train.txt", args.train_bytes, args.raw_dir)
    val_raw = download_prefix("TinyStories-valid.txt", args.val_bytes, args.raw_dir)
    train_stories, train_keys = distinct_stories(stories_from_prefix(train_raw))
    val_stories, _ = distinct_stories(stories_from_prefix(val_raw), train_keys)
    if not train_stories or not val_stories:
        raise ValueError("Not enough complete, distinct stories; increase download sizes")
    splits = {"train": "\n\n".join(train_stories) + "\n",
              "val": "\n\n".join(val_stories) + "\n"}
    # A fixed ASCII alphabet makes prompts predictable without inspecting validation.
    tokenizer = CharTokenizer(["\n"] + [chr(c) for c in range(32, 127)])
    manifest = {
        "dataset": DATASET, "revision": REVISION,
        "dataset_url": f"https://huggingface.co/datasets/{DATASET}",
        "license": "CDLA-Sharing-1.0",
        "license_url": "https://cdla.dev/sharing-1-0/",
        "selection": "Deterministic prefixes of the official train and validation files; not random samples.",
        "normalization": "ASCII with normalized whitespace and punctuation; complete stories only; exact normalized story deduplication.",
        "raw_sha256": {"train": digest(train_raw), "val": digest(val_raw)},
        "download_bytes": {"train": len(train_raw), "val": len(val_raw)},
        "splits": {}, "vocab_size": len(tokenizer.chars),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, text in splits.items():
        tokenizer.encode(text)
        data = text.encode("utf-8")
        (args.output / f"{name}.txt").write_bytes(data)
        manifest["splits"][name] = {
            "sha256": digest(data), "characters": len(text),
            "stories": len(train_stories if name == "train" else val_stories),
        }
    (args.output / "tokenizer.json").write_text(json.dumps(tokenizer.chars) + "\n")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["splits"], indent=2))
    print(f"Prepared data in {args.output}; vocabulary: {len(tokenizer.chars)} characters.")


if __name__ == "__main__":
    main()
