"""Fit a reversible word vocabulary using training text only."""

import argparse
import json
from pathlib import Path

from joi.prepare import digest
from joi.tokenizer import WordTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("data/processed/dialogue-wide"))
    parser.add_argument("--output",type=Path,default=Path("data/processed/dialogue-words"))
    parser.add_argument("--vocab-size",type=int,default=2048)
    args = parser.parse_args()
    train = (args.source/"train.txt").read_text()
    tokenizer = WordTokenizer.from_text(train,args.vocab_size)
    manifest = json.loads((args.source/"manifest.json").read_text())
    manifest["parent_manifest_sha256"] = digest((args.source/"manifest.json").read_bytes())
    manifest["tokenizer"] = {"type":"word", "vocab_size":len(tokenizer.chars),
                             "fit":"training text only; frequent words plus fixed ASCII fallback"}
    manifest["vocab_size"] = len(tokenizer.chars)
    args.output.mkdir(parents=True,exist_ok=True)
    for split in ("train","val"):
        raw = (args.source/f"{split}.txt").read_bytes()
        if digest(raw) != manifest["splits"][split]["sha256"]:
            raise ValueError("Source text does not match its manifest")
        text = raw.decode()
        encoded = tokenizer.encode(text)
        if tokenizer.decode(encoded) != text:
            raise ValueError("Tokenizer failed lossless roundtrip")
        (args.output/f"{split}.txt").write_bytes(raw)
        manifest["splits"][split]["tokens"] = len(encoded)
    (args.output/"tokenizer.json").write_text(json.dumps({"type":"word","pieces":tokenizer.chars})+"\n")
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"vocabulary":len(tokenizer.chars),"splits":manifest["splits"]},indent=2))


if __name__ == "__main__":
    main()
