import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from joi.checkpoint import save_checkpoint
from joi.compare import compare_reports
from joi.evaluate import evaluate_checkpoint
from joi.model import Joi, ModelConfig
from joi.tokenizer import CharTokenizer


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_checkpoint_self_comparison_is_reproducible_and_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = CharTokenizer.from_text("abc\n")
            (root / "tokenizer.json").write_text(json.dumps(tokenizer.chars))
            hashes, manifest = {}, {"splits": {}}
            for name, text in {"train": "abc\n" * 50, "val": "bca\n" * 20}.items():
                raw = text.encode()
                (root / f"{name}.txt").write_bytes(raw)
                hashes[name] = hashlib.sha256(raw).hexdigest()
                manifest["splits"][name] = {"sha256": hashes[name]}
            (root / "manifest.json").write_text(json.dumps(manifest))
            prompts = root / "prompts.json"
            prompts.write_text(json.dumps([{"purpose": "test", "prompt": "ab"}]))
            model = Joi(ModelConfig(vocab_size=4, block_size=8, n_embd=8, n_heads=2,
                                    n_layers=1, dropout=0.1))
            checkpoint = root / "model.pt"
            save_checkpoint(checkpoint, model, tokenizer, step=3, data_hashes=hashes)
            with patch("socket.socket", side_effect=AssertionError("Network access attempted")):
                first = evaluate_checkpoint(checkpoint, root, prompts, batches=2, tokens=12)
                torch.manual_seed(999)
                second = evaluate_checkpoint(checkpoint, root, prompts, batches=2, tokens=12)
            self.assertEqual(first, second)
            comparison = compare_reports(first, second)
            self.assertEqual(comparison["loss_change_nats"], 0)
            self.assertEqual(comparison["lower_loss_checkpoint"], "tie")
            self.assertEqual(len(first["samples"][0]["continuation"]), 12)
            for field in ("dataset", "vocabulary_sha256", "evaluation", "generation", "torch_version"):
                changed = copy.deepcopy(second)
                changed[field] = "a different protocol"
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Cannot compare"):
                    compare_reports(first, changed)
            changed = copy.deepcopy(second)
            changed["samples"][0]["prompt"] = "ca"
            with self.assertRaisesRegex(ValueError, "different prompts"):
                compare_reports(first, changed)
            changed = copy.deepcopy(second)
            changed["validation_loss_nats"] = first["validation_loss_nats"] * 0.8
            result = compare_reports(first, changed)
            self.assertAlmostEqual(result["relative_loss_reduction_percent"], 20)
            self.assertEqual(result["lower_loss_checkpoint"], "candidate")


if __name__ == "__main__":
    unittest.main()
