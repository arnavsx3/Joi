import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from joi.checkpoint import save_checkpoint
from joi.chat_evaluate import evaluate_chat, literal_recall
from joi.model import Joi, ModelConfig
from joi.tokenizer import CharTokenizer


class ChatEvaluationTests(unittest.TestCase):
    def test_partial_name_is_not_counted_as_correct_recall(self):
        self.assertTrue(literal_recall("Neel","Your name is Neel."))
        self.assertFalse(literal_recall("Neel","Your name is Neell."))

    def test_real_checkpoint_probe_is_reproducible_and_records_generated_history(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        tokenizer = CharTokenizer(["\n"] + [chr(c) for c in range(32,127)])
        model = Joi(ModelConfig(vocab_size=96,block_size=192,n_embd=8,n_heads=2,n_layers=1,dropout=0))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"format":"joi-dialogue-v1","splits":{}}
            hashes = {}
            for split in ("train","val"):
                raw = ("Persona: cheerful, warm, playful, curious.\nUser: Hi!\nJoi: Hello!\n\n" * 10).encode()
                (root/f"{split}.txt").write_bytes(raw)
                hashes[split]=hashlib.sha256(raw).hexdigest()
                manifest["splits"][split]={"sha256":hashes[split]}
            (root/"tokenizer.json").write_text(json.dumps(tokenizer.chars))
            (root/"manifest.json").write_text(json.dumps(manifest))
            probes = root/"probes.json"
            probes.write_text(json.dumps([{"id":"two-turn","turns":["Hi!","How are you?"]}]))
            checkpoint = root/"model.pt"
            save_checkpoint(checkpoint,model,tokenizer,step=2,data_hashes=hashes,dataset_manifest=manifest)
            first = evaluate_chat(checkpoint,root,probes,tokens=8,batches=1)
            second = evaluate_chat(checkpoint,root,probes,tokens=8,batches=1)
            self.assertEqual(first["checkpoint_sha256"],second["checkpoint_sha256"])
            self.assertEqual(first["validation_loss_nats"],second["validation_loss_nats"])
            left = first["episodes"][0]["results"]
            right = second["episodes"][0]["results"]
            self.assertEqual(len(left),2)
            for before,after in zip(left,right):
                self.assertEqual(before["reply"],after["reply"])
            self.assertEqual(left[0]["included_messages"],1)
            self.assertEqual(left[1]["included_messages"],3)


if __name__ == "__main__":
    unittest.main()
