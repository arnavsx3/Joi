"""Optional spelling constraints learned from our training text, not canned replies."""

import re

WORDS = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
ENDING = re.compile(r"[A-Za-z]+(?:'[A-Za-z]*)?$")
LETTERS = re.compile(r"[A-Za-z']+\Z")


def training_words(text):
    return sorted({match.group().lower() for match in WORDS.finditer(text)})


class WordBoundaryConstraint:
    """Forbid joining tokens into spellings absent from training or the prompt.

    Every allowed token is still selected from the model's probability
    distribution. This neither selects a topic nor supplies answer text. Words
    in the prompt allow unfamiliar names; no stored answer/name is substituted.
    Training text can itself contain misspellings, so this is not a dictionary.
    """
    def __init__(self, tokenizer, words, prompt=""):
        self.tokenizer = tokenizer
        self.words = set(words) | set(training_words(prompt))
        self.prefixes = {word[:index] for word in self.words for index in range(1, len(word) + 1)}
        self.cache = {}

    def __call__(self, generated):
        text = self.tokenizer.decode(generated)
        ending = ENDING.search(text)
        prefix = ending.group().lower() if ending else ""
        if prefix not in self.cache:
            allowed = []
            for index, piece in enumerate(self.tokenizer.chars):
                if LETTERS.fullmatch(piece) and (piece != "'" or prefix):
                    valid = prefix + piece.lower() in self.prefixes
                else:
                    valid = not prefix or prefix in self.words
                if valid:
                    allowed.append(index)
            self.cache[prefix] = allowed
        return self.cache[prefix]
