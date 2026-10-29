"""A local conversation engine backed only by Joi's own learned weights."""

import argparse
from pathlib import Path
import threading
import time

import torch

from joi.checkpoint import load_checkpoint
from joi.dialogue import build_prompt


class ChatEngine:
    def __init__(self, checkpoint, *, threads=2, tokens=180, temperature=0.65, top_k=15):
        torch.set_num_threads(threads)
        self.model, self.tokenizer, self.checkpoint = load_checkpoint(Path(checkpoint))
        if self.checkpoint.get("dataset_manifest", {}).get("format") != "joi-dialogue-v1":
            raise ValueError("Select a dialogue-trained checkpoint; prepare and train dialogue first")
        self.model.eval()
        self.tokens, self.temperature, self.top_k = tokens, temperature, top_k
        self.lock = threading.Lock()

    def reply(self, messages, *, seed=42):
        prompt, included = build_prompt(messages, self.model.config.block_size)
        encoded = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
        started = time.monotonic()
        with self.lock:
            torch.manual_seed(seed)
            output = self.model.generate(encoded, max_new_tokens=self.tokens,
                                         temperature=self.temperature, top_k=self.top_k,
                                         stop_sequences=[self.tokenizer.encode("\n")])
        reply = self.tokenizer.decode(output[0, encoded.shape[1]:].tolist()).strip()
        if not reply:
            raise ValueError("Joi didn't finish a reply this time. Please try again.")
        return {"reply":reply, "included_messages":included,
                "context_characters":len(prompt), "elapsed_seconds":round(time.monotonic()-started,3)}

    def status(self):
        return {"name":"Joi", "step":self.checkpoint.get("step"),
                "context_size":self.model.config.block_size,
                "parameters":sum(parameter.numel() for parameter in self.model.parameters()),
                "mode":"our-own-dialogue-model", "training_format":"joi-dialogue-v1"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/dialogue-cpu/best.pt"))
    args = parser.parse_args()
    try:
        engine = ChatEngine(args.checkpoint)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    messages = []
    print("Joi · local chat. /quit leaves, /new starts a fresh conversation.")
    while True:
        try:
            text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSee you soon.")
            return
        if text == "/quit":
            return
        if text == "/new":
            messages.clear()
            continue
        if not text:
            continue
        candidate = messages + [{"role":"user", "content":text}]
        try:
            result = engine.reply(candidate, seed=42 + len(messages))
        except ValueError as exc:
            print(str(exc))
            continue
        print("Joi> " + result["reply"])
        messages = candidate + [{"role":"assistant", "content":result["reply"]}]


if __name__ == "__main__":
    main()
