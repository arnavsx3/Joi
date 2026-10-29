"""A transparent, character-level vocabulary. No pretrained tokenizer."""

from collections import Counter
import re

PIECES = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^A-Za-z0-9]")


class CharTokenizer:
    kind = "char"
    def __init__(self, chars):
        self.chars = list(chars)
        if not self.chars or len(self.chars) != len(set(self.chars)):
            raise ValueError("Vocabulary must be nonempty and contain unique characters")
        if any(len(c) != 1 for c in self.chars):
            raise ValueError("Each vocabulary entry must be one character")
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    def encode(self, text):
        unknown = sorted(set(text) - self.stoi.keys())
        if unknown:
            raise ValueError(f"Characters not in training vocabulary: {unknown!r}")
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.chars[int(i)] for i in ids)


class WordTokenizer:
    """Learn common whole-word units; unseen words fall back to ASCII characters."""
    kind = "word"

    def __init__(self, pieces):
        self.chars = list(pieces)  # Historical checkpoint vocabulary field.
        if len(set(self.chars)) != len(self.chars) or any(not piece for piece in self.chars):
            raise ValueError("Vocabulary pieces must be nonempty and unique")
        self.stoi = {piece:index for index,piece in enumerate(self.chars)}
        if any(character not in self.stoi for character in ["\n"]+[chr(c) for c in range(32,127)]):
            raise ValueError("Word vocabulary must include the fixed ASCII fallback alphabet")

    @classmethod
    def from_text(cls, text, vocab_size=2048):
        alphabet = ["\n"]+[chr(c) for c in range(32,127)]
        if vocab_size < len(alphabet):
            raise ValueError("Vocabulary must fit the ASCII fallback alphabet")
        counts = Counter(piece for piece in PIECES.findall(text) if len(piece)>1)
        words = sorted(counts, key=lambda piece:(-counts[piece],piece))[:vocab_size-len(alphabet)]
        return cls(alphabet+words)

    def encode(self, text):
        result = []
        for piece in PIECES.findall(text):
            if piece in self.stoi:
                result.append(self.stoi[piece])
            else:
                try:
                    result.extend(self.stoi[character] for character in piece)
                except KeyError as exc:
                    raise ValueError(f"Character outside the ASCII vocabulary: {exc.args[0]!r}") from exc
        return result

    def decode(self, ids):
        return "".join(self.chars[int(index)] for index in ids)


def tokenizer_from_state(state, kind=None):
    if isinstance(state, dict):
        kind, state = state["type"], state["pieces"]
    if kind in (None,"char"):
        return CharTokenizer(state)
    if kind == "word":
        return WordTokenizer(state)
    raise ValueError(f"Unknown tokenizer type: {kind}")
