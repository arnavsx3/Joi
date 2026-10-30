"""A local conversation engine backed only by Joi's own learned weights."""

import argparse
import hashlib
import io
import re
from pathlib import Path
import threading
import time

import torch

from joi.checkpoint import load_checkpoint
from joi.dialogue import build_prompt, clean_turn
from joi.lexical import WordBoundaryConstraint


class ChatEngine:
    def __init__(self, checkpoint, *, threads=2, tokens=None, temperature=None, top_k=None, beams=None):
        torch.set_num_threads(threads)
        raw = Path(checkpoint).read_bytes()
        self.model, self.tokenizer, self.checkpoint = load_checkpoint(io.BytesIO(raw))
        self.checkpoint_sha256 = hashlib.sha256(raw).hexdigest()
        if self.checkpoint.get("dataset_manifest", {}).get("format") != "joi-dialogue-v1":
            raise ValueError("Select a dialogue-trained checkpoint; prepare and train dialogue first")
        self.model.eval()
        decoding = self.checkpoint.get("decoding_settings", {})
        self.tokens = tokens if tokens is not None else decoding.get("max_new_tokens", 96 if self.tokenizer.kind=="word" else 180)
        self.temperature = temperature if temperature is not None else decoding.get("temperature", 0.65)
        self.top_k = top_k if top_k is not None else decoding.get("top_k", 15)
        self.beams = beams if beams is not None else decoding.get("beams", 1)
        # Spaces are tokens too, so six raw tokens cover roughly three words.
        # Blocking three raw tokens would forbid ordinary repeated words.
        self.no_repeat_ngram_size = 6 if self.tokenizer.kind=="word" else 0
        self.lock = threading.Lock()
        self.lexical_words = self.checkpoint.get("lexical_words") if self.tokenizer.kind == "word" else None

    def reply(self, messages, *, seed=42):
        prompt, included = build_prompt(messages, self.model.config.block_size, self.tokenizer)
        encoded = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
        started = time.monotonic()
        constraint = WordBoundaryConstraint(self.tokenizer, self.lexical_words, prompt) if self.lexical_words else None
        with self.lock:
            torch.manual_seed(seed)
            output = self.model.generate(encoded, max_new_tokens=self.tokens,
                                         temperature=self.temperature, top_k=self.top_k,
                                         no_repeat_ngram_size=self.no_repeat_ngram_size,
                                         allowed_tokens_fn=constraint,
                                         num_beams=self.beams,
                                         stop_sequences=[self.tokenizer.encode("\n")])
        reply = self.tokenizer.decode(output[0, encoded.shape[1]:].tolist()).strip()
        if not reply:
            raise ValueError("Joi didn't finish a reply this time. Please try again.")
        generated_tokens = output.shape[1] - encoded.shape[1]
        return {"reply":reply, "included_messages":included, "generated_tokens":generated_tokens,
                "stopped_at_turn_boundary":generated_tokens < self.tokens,
                "context_characters":len(prompt), "elapsed_seconds":round(time.monotonic()-started,3)}

    def status(self):
        return {"name":"Joi", "step":self.checkpoint.get("step"),
                "context_size":self.model.config.block_size,
                "parameters":sum(parameter.numel() for parameter in self.model.parameters()),
                "tokenizer":self.tokenizer.kind,
                "word_boundary_constraint":bool(self.lexical_words),
                "temperature":self.temperature, "top_k":self.top_k, "max_new_tokens":self.tokens, "beams":self.beams,
                "mode":"our-own-dialogue-model", "training_format":"joi-dialogue-v1"}


class CompanionEngine:
    """Use our general model for chat and a compatible neural recall specialist."""
    def __init__(self, checkpoint, *, use_recall=True, tokens=None, temperature=None, top_k=None, beams=None):
        self.general=ChatEngine(checkpoint,tokens=tokens,temperature=temperature,top_k=top_k,beams=beams)
        self.recall=None
        recall_path=Path(__file__).resolve().parent.parent/"checkpoints/companion-recall/best.pt"
        if use_recall and self.general.tokenizer.kind=="word" and recall_path.is_file():
            candidate=ChatEngine(recall_path,tokens=tokens if tokens is not None else self.general.tokens,
                                 temperature=self.general.temperature,top_k=self.general.top_k,beams=self.general.beams)
            # Each network encodes the shared text with its own vocabulary.
            # Vocabulary growth in the general model need not discard our older
            # neural recall model; token IDs never cross between networks.
            if candidate.tokenizer.kind=="word":
                candidate.lock=self.general.lock
                candidate.lexical_words=self.general.lexical_words
                self.recall=candidate
        for name in ("model","tokenizer","checkpoint","checkpoint_sha256","no_repeat_ngram_size"):
            setattr(self,name,getattr(self.general,name))

    def uses_recall(self,messages):
        if not self.recall or not messages:
            return False
        question=clean_turn(messages[-1].get("content","")).lower()
        if not re.search(r"(?:what.*\bmy name\b|remember.*\bmy name\b|recall.*\bmy name\b|say.*\bmy name\b)",question):
            return False
        for message in reversed(messages[:-1]):
            if message.get("role")!="user":
                continue
            text=clean_turn(message.get("content",""))
            match=re.search(r"(?:my name is|you can call me|call me)\s+([A-Za-z]+)",text,flags=re.I)
            if not match:
                match=re.search(r"I'm\s+([A-Za-z]+),?\s+by the way",text,flags=re.I)
            if match:
                # Known whole-word names are better handled by the general
                # model; the specialist learned character-fallback names.
                return match.group(1) not in self.general.tokenizer.stoi
        return False

    def reply(self,messages,*,seed=42):
        specialist=self.uses_recall(messages)
        engine=self.recall if specialist else self.general
        return {**engine.reply(messages,seed=seed),
                "model_used":"neural-recall-specialist" if specialist else "neural-general-dialogue"}

    def status(self):
        return {**self.general.status(),"recall_specialist":self.recall is not None,
                "models":2 if self.recall else 1,
                "recall_checkpoint_sha256":self.recall.checkpoint_sha256 if self.recall else None,
                "recall_word_boundary_constraint":bool(self.recall and self.recall.lexical_words),
                "note":"Routing selects a locally trained neural model. Names and reply text are not substituted by a template."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/joi-chat/best.pt"))
    parser.add_argument("--no-recall-specialist",action="store_true")
    args = parser.parse_args()
    try:
        engine = CompanionEngine(args.checkpoint,use_recall=not args.no_recall_specialist)
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
