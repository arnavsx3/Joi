import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from joi.checkpoint import load_checkpoint, save_checkpoint
from joi.model import Joi, ModelConfig
from joi.prepare import distinct_stories, stories_from_prefix
from joi.tokenizer import CharTokenizer


class JoiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)

    def test_tokenizer_roundtrip_and_unknown_character(self):
        tokenizer = CharTokenizer.from_text("Hello, Joi!\n")
        self.assertEqual(tokenizer.decode(tokenizer.encode("Joi!\n")), "Joi!\n")
        with self.assertRaisesRegex(ValueError, "not in training"):
            tokenizer.encode("\u2603")

    def test_future_characters_cannot_change_past_predictions(self):
        model = Joi(ModelConfig(vocab_size=12, block_size=8, n_embd=16, n_heads=2, n_layers=2))
        model.eval()
        original = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        changed = torch.tensor([[1, 2, 3, 9, 9, 9, 9, 9]])
        with torch.no_grad():
            left, _ = model(original)
            right, _ = model(changed)
        torch.testing.assert_close(left[:, :3], right[:, :3], rtol=0, atol=0)
        self.assertFalse(torch.allclose(left[:, 3:], right[:, 3:]))

    def test_model_can_learn_a_simple_sequence(self):
        model = Joi(ModelConfig(vocab_size=3, block_size=9, n_embd=16, n_heads=2,
                                n_layers=1, dropout=0))
        x = torch.tensor([[0, 1, 2] * 3] * 4)
        y = torch.tensor([[1, 2, 0] * 3] * 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        initial = model(x, y)[1].item()
        for _ in range(50):
            optimizer.zero_grad(set_to_none=True)
            loss = model(x, y)[1]
            loss.backward()
            optimizer.step()
        self.assertLess(model(x, y)[1].item(), initial / 5)

    def test_preparation_discards_partial_stories_and_split_duplicates(self):
        raw = 'One nice story.<|endoftext|>One nice story.<|endoftext|>An unfinished'.encode()
        stories = stories_from_prefix(raw)
        self.assertEqual(len(stories), 2)
        train, seen = distinct_stories(stories)
        validation, _ = distinct_stories(["One nice story.", "A new story."], seen)
        self.assertEqual(train, ["One nice story."])
        self.assertEqual(validation, ["A new story."])

    def test_checkpoint_reload_and_generation_are_offline(self):
        tokenizer = CharTokenizer.from_text("abc")
        model = Joi(ModelConfig(vocab_size=3, block_size=4, n_embd=8, n_heads=2,
                                n_layers=1, dropout=0))
        model.eval()
        prompt = torch.tensor([[0, 1]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.pt"
            save_checkpoint(path, model, tokenizer, step=1)
            with patch("socket.socket", side_effect=AssertionError("Network access attempted")):
                restored, restored_tokenizer, payload = load_checkpoint(path)
                restored.eval()
                torch.testing.assert_close(model(prompt)[0], restored(prompt)[0], rtol=0, atol=0)
                output = restored.generate(prompt, max_new_tokens=10)
            self.assertEqual(output.shape, (1, 12))
            self.assertEqual(restored_tokenizer.chars, tokenizer.chars)
            self.assertEqual(payload["step"], 1)

    def test_resume_matches_uninterrupted_training_with_dropout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            tokenizer = CharTokenizer.from_text("abc\n")
            (data / "tokenizer.json").write_text(json.dumps(tokenizer.chars))
            manifest = {"splits": {}}
            for split, text in {"train": "abc\n" * 100, "val": "bc\na" * 40}.items():
                raw = text.encode()
                (data / f"{split}.txt").write_bytes(raw)
                manifest["splits"][split] = {"sha256": hashlib.sha256(raw).hexdigest()}
            (data / "manifest.json").write_text(json.dumps(manifest))
            config = root / "config.json"
            config.write_text(json.dumps({"block_size": 8, "n_embd": 8, "n_heads": 2,
                                          "n_layers": 1, "dropout": 0.2}))
            base = [sys.executable, "-m", "joi.train", "--data", str(data), "--config", str(config),
                    "--batch-size", "2", "--eval-batches", "1", "--eval-every", "2", "--threads", "1"]

            def run(*args):
                result = subprocess.run(base + list(args), capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            full, resumed = root / "full", root / "resumed"
            run("--out", str(full), "--steps", "6")
            run("--out", str(resumed), "--steps", "4")
            run("--out", str(resumed), "--resume", str(resumed / "last.pt"), "--steps", "6")
            _, _, left = load_checkpoint(full / "last.pt")
            _, _, right = load_checkpoint(resumed / "last.pt")
            for key in left["model_state"]:
                torch.testing.assert_close(left["model_state"][key], right["model_state"][key], rtol=0, atol=0)
            self.assertTrue(torch.equal(left["torch_rng_state"], right["torch_rng_state"]))
            self.assertTrue(torch.equal(left["batch_rng_state"], right["batch_rng_state"]))
            self.assertEqual(left["metrics"][-1]["val_loss"], right["metrics"][-1]["val_loss"])


if __name__ == "__main__":
    unittest.main()
