import unittest
from unittest.mock import patch

import torch

from joi.generate import generate_text
from joi.model import Joi, ModelConfig
from joi.play import run_session
from joi.tokenizer import CharTokenizer


class PlaygroundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_session_recovers_from_invalid_text_and_keeps_weights_fixed(self):
        torch.manual_seed(1)
        tokenizer = CharTokenizer.from_text("abc\n")
        model = Joi(ModelConfig(vocab_size=4, block_size=8, n_embd=8, n_heads=2,
                                n_layers=1, dropout=0.1))
        weights = {key: value.clone() for key, value in model.state_dict().items()}
        expected = [generate_text(model, tokenizer, prompt, tokens=12, seed=seed)[len(prompt):]
                    for prompt, seed in (("ab", 7), ("bc", 8))]
        prompts = iter(["", "\u2603", "ab", "/help", "bc", "/quit"])
        output = []
        with patch("socket.socket", side_effect=AssertionError("Network access attempted")):
            run_session(model, tokenizer, tokens=12, seed=7,
                        input_fn=lambda _: next(prompts), output_fn=output.append)
        self.assertEqual([line[5:] for line in output if line.startswith("Joi> ")], expected)
        self.assertTrue(any("Please try again" in line for line in output))
        self.assertEqual(output[-1], "Session ended.")
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)

    def test_eof_ends_session(self):
        output = []
        def eof(_):
            raise EOFError
        run_session(None, None, input_fn=eof, output_fn=output.append)
        self.assertEqual(output[-1], "\nSession ended.")


if __name__ == "__main__":
    unittest.main()
