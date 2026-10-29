from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch

from joi.checkpoint import save_checkpoint
from joi.dialogue import build_prompt, clean_turn, format_conversation
from joi.model import Joi, ModelConfig
from joi.tokenizer import CharTokenizer
from joi.train import initialize_from_checkpoint


class DialogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_context_keeps_recent_complete_turns_and_persona(self):
        messages = [{"role":"user", "content":"My name is Arnav."},
                    {"role":"assistant", "content":"Hi, Arnav!"},
                    {"role":"user", "content":"What is my name?"}]
        prompt, count = build_prompt(messages, 384)
        self.assertEqual(count, 3)
        self.assertIn("My name is Arnav.", prompt)
        self.assertTrue(prompt.endswith("User: What is my name?\nJoi: "))
        long_history = [{"role":"user", "content":"old " * 50},
                        {"role":"assistant", "content":"old reply " * 30}] + messages
        prompt, count = build_prompt(long_history, 384)
        self.assertEqual(count, 3)
        self.assertNotIn("old", prompt)
        self.assertTrue(prompt.startswith("Persona:"))
        with self.assertRaisesRegex(ValueError, "little long"):
            build_prompt([{"role":"user", "content":"x" * 600}], 384)
        with self.assertRaisesRegex(ValueError, "end with"):
            build_prompt(messages[:-1], 384)

    def test_dialogue_normalization_removes_spurious_punctuation_spacing(self):
        self.assertEqual(clean_turn("I ' m here . Let ' s talk !"), "I'm here. Let's talk!")
        self.assertEqual(clean_turn("Hey\nthere!"), "Hey there!")
        self.assertIn("Joi: Hi!\n", format_conversation([{"role":"assistant", "content":"Hi !"}]))

    def test_context_expansion_preserves_learned_weights_and_original_predictions(self):
        torch.manual_seed(4)
        tokenizer = CharTokenizer.from_text("abc")
        source = Joi(ModelConfig(vocab_size=3, block_size=8, n_embd=8,
                                 n_heads=2, n_layers=1, dropout=0.1))
        source.eval()
        config = ModelConfig(**{**asdict(source.config), "block_size":16})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            save_checkpoint(path, source, tokenizer, step=7)
            target, ancestry = initialize_from_checkpoint(path, config, tokenizer)
            target.eval()
            tokens = torch.tensor([[0,1,2,0,1,2]])
            torch.testing.assert_close(source(tokens)[0], target(tokens)[0], rtol=0, atol=0)
            torch.testing.assert_close(source.position_embedding.weight,
                                       target.position_embedding.weight[:8], rtol=0, atol=0)
            self.assertEqual(ancestry["new_positions"], 8)
            self.assertEqual(ancestry["source_step"], 7)
            self.assertEqual(len(ancestry["source_sha256"]),64)


if __name__ == "__main__":
    unittest.main()
