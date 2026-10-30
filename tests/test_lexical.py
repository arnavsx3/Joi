import unittest
from unittest.mock import patch

import torch

from joi.lexical import WordBoundaryConstraint, training_words
from joi.model import Joi, ModelConfig
from joi.tokenizer import WordTokenizer


class LexicalTests(unittest.TestCase):
    def test_blocks_mangled_joins_and_allows_unfamiliar_prompt_name(self):
        tokenizer = WordTokenizer.from_text("tea tea talk talk", 105)
        constraint = WordBoundaryConstraint(tokenizer, ["tea", "talk", "don't"], "My name is Neel.")
        allowed = lambda prefix, piece: tokenizer.stoi[piece] in constraint(tokenizer.encode(prefix))
        self.assertTrue(allowed("tea", " "))
        self.assertFalse(allowed("tea", "talk"))
        self.assertFalse(allowed("tal", " "))
        self.assertTrue(allowed("tal", "k"))
        self.assertTrue(allowed("Nee", "l"))
        self.assertTrue(allowed("Neel", "."))
        self.assertTrue(allowed("don", "'"))
        self.assertTrue(allowed("don't", " "))
        self.assertFalse(allowed("te", "\n"))

    def test_constraint_masks_even_the_highest_probability_invalid_token(self):
        torch.set_num_threads(1)
        model = Joi(ModelConfig(vocab_size=5, block_size=8, n_embd=8, n_heads=2, n_layers=1, dropout=0))
        def choose(probabilities, **kwargs):
            self.assertEqual(probabilities[0, 0].item(), 0)
            self.assertEqual(probabilities[0, 1].item(), 1)
            return torch.tensor([[1]])
        with patch("torch.multinomial", side_effect=choose):
            generated = model.generate(torch.tensor([[0]]), max_new_tokens=1, allowed_tokens_fn=lambda _: [1])
        self.assertEqual(generated.tolist(), [[0, 1]])
        with self.assertRaisesRegex(ValueError, "No valid next token"):
            model.generate(torch.tensor([[0]]), max_new_tokens=1, allowed_tokens_fn=lambda _: [])


if __name__ == "__main__":
    unittest.main()
