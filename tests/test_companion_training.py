import random
import unittest

import torch

from joi.companion_data import records, prepare_examples
from joi.model import Joi, ModelConfig
from joi.tokenizer import WordTokenizer
from joi.tune_companion import contrasting_batch, per_example_loss


class CompanionTrainingTests(unittest.TestCase):
    def test_user_phrasings_and_memory_values_are_separate(self):
        train, validation = records("train"), records("validation")
        # Prelude greetings and memory question formats are intentionally shared.
        # Situation user phrasings and memory introductions/values are held out.
        users = lambda rows: {m["content"] for topic, history, _ in rows if not topic.startswith("memory-") for m in history
                              if m["role"] == "user" and m["content"] != "Hello!"}
        self.assertFalse(users(train) & users(validation))
        self.assertTrue(any("purple" in target for _, _, target in validation))
        self.assertFalse(any("purple" in target for _, _, target in train))

    def test_contrast_preserves_answer_but_changes_context_and_padding_is_ignored(self):
        torch.set_num_threads(1)
        rows = [("tea", [{"role": "user", "content": "I like tea."}], "Tea is nice."),
                ("code", [{"role": "user", "content": "I write code."}], "What are you making?")]
        tokenizer = WordTokenizer.from_text("I like tea Tea is nice write code What are you making Persona cheerful warm playful curious User Joi", 150)
        examples = prepare_examples(tokenizer, 128, rows)
        positive, negative = contrasting_batch(examples, 4, random.Random(5), 128)
        for index in range(4):
            p = positive[1][index][positive[1][index] != -100]
            n = negative[1][index][negative[1][index] != -100]
            self.assertEqual(p.tolist(), n.tolist())
            p_prefix = positive[0][index][positive[2][index] == 0]
            n_prefix = negative[0][index][negative[2][index] == 0]
            self.assertNotEqual(p_prefix.tolist(), n_prefix.tolist())
        model = Joi(ModelConfig(len(tokenizer.chars), block_size=128, n_embd=8, n_heads=2, n_layers=1, dropout=0))
        losses = per_example_loss(model, positive)
        self.assertEqual(losses.shape, (4,))
        self.assertTrue(torch.isfinite(losses).all())
        altered = tuple(part.clone() for part in positive)
        altered[0][altered[1] == -100] = 0
        # Changing right padding cannot change supervised predictions before it.
        lengths = [(y != -100).nonzero()[-1].item() + 1 for y in positive[1]]
        for index, length in enumerate(lengths):
            altered[0][index, :length] = positive[0][index, :length]
        torch.testing.assert_close(losses, per_example_loss(model, altered))
        losses.mean().backward()
        self.assertGreater(model.token_embedding.weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
