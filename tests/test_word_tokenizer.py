from pathlib import Path
import tempfile
import unittest

import torch

from joi.checkpoint import load_checkpoint, save_checkpoint
from joi.dialogue import build_prompt
from joi.model import Joi, ModelConfig
from joi.tokenizer import WordTokenizer, tokenizer_from_state


class WordTokenizerTests(unittest.TestCase):
    def test_vocabulary_is_deterministic_and_unseen_words_roundtrip(self):
        text = "Hello friend! Hello friend! I'm Joi.\n"
        tokenizer = WordTokenizer.from_text(text,102)
        self.assertEqual(tokenizer.chars,WordTokenizer.from_text(text,102).chars)
        self.assertIn("friend",tokenizer.stoi)
        unseen = "A new name is Neel. 42!\n"
        self.assertEqual(tokenizer.decode(tokenizer.encode(unseen)),unseen)
        self.assertNotIn("Neel",tokenizer.stoi)
        self.assertLess(len(tokenizer.encode(text)),len(text))
        restored = tokenizer_from_state({"type":"word","pieces":tokenizer.chars})
        self.assertEqual(restored.encode(unseen),tokenizer.encode(unseen))

    def test_checkpoint_loads_its_learned_word_vocabulary(self):
        tokenizer = WordTokenizer.from_text("Hello friend. I'm Joi. "*5,110)
        model = Joi(ModelConfig(vocab_size=len(tokenizer.chars),block_size=64,
                                n_embd=8,n_heads=2,n_layers=1,dropout=0))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"word.pt"
            save_checkpoint(path,model,tokenizer,step=1)
            restored, vocabulary, _ = load_checkpoint(path)
            self.assertEqual(vocabulary.kind,"word")
            self.assertEqual(vocabulary.chars,tokenizer.chars)
            tokens=torch.tensor([tokenizer.encode("Hello friend.")])
            torch.testing.assert_close(model(tokens)[0],restored(tokens)[0],rtol=0,atol=0)

    def test_prompt_budget_uses_tokens_rather_than_characters(self):
        text="Persona cheerful warm playful curious User Joi Hello friend name Arnav What is my "*4
        tokenizer=WordTokenizer.from_text(text,130)
        messages=[{"role":"user","content":"My name is Arnav."},
                  {"role":"assistant","content":"Hello friend!"},
                  {"role":"user","content":"What is my name?"}]
        prompt,count=build_prompt(messages,128,tokenizer)
        self.assertEqual(count,3)
        self.assertLessEqual(len(tokenizer.encode(prompt)),96)


if __name__ == "__main__":
    unittest.main()
