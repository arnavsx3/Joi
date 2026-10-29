"""A transparent, character-level vocabulary. No pretrained tokenizer."""


class CharTokenizer:
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
