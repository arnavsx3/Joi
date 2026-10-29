from http.server import ThreadingHTTPServer
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

import torch

from joi.model import Joi, ModelConfig
from joi.serve import handler_for


class StopSequenceTests(unittest.TestCase):
    def test_repetition_blocking_applies_only_to_generated_patterns(self):
        model = Joi(ModelConfig(vocab_size=5,block_size=16,n_embd=8,n_heads=2,n_layers=1,dropout=0))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        def choose(probabilities, **kwargs):
            token = next(index for index in (1,2,3,4,0) if probabilities[0,index] > 0)
            return torch.tensor([[token]])
        with patch("torch.multinomial",side_effect=choose):
            result=model.generate(torch.tensor([[1,1]]),max_new_tokens=5,no_repeat_ngram_size=2)
        generated=result[0,2:].tolist()
        self.assertEqual(generated[:2],[1,1])  # Prompt patterns do not forbid copying a name.
        pairs=list(zip(generated,generated[1:]))
        self.assertEqual(len(pairs),len(set(pairs)))

    def test_generated_turn_marker_is_removed_without_generating_another_speaker(self):
        torch.set_num_threads(1)
        model = Joi(ModelConfig(vocab_size=5, block_size=8, n_embd=8,
                                n_heads=2, n_layers=1, dropout=0))
        model.train()
        with patch("torch.multinomial", side_effect=[torch.tensor([[1]]), torch.tensor([[2]]), torch.tensor([[3]])]):
            result = model.generate(torch.tensor([[0]]), max_new_tokens=10, stop_sequences=[[2,3]])
        self.assertEqual(result.tolist(), [[0,1]])
        self.assertTrue(model.training)


class LocalChatApiTests(unittest.TestCase):
    def setUp(self):
        class Engine:
            received = None
            def status(self):
                return {"name":"Joi", "step":200, "context_size":384, "parameters":458240}
            def reply(self, messages, **kwargs):
                self.received = messages
                return {"reply":"Hi, Arnav! What are you making?", "included_messages":len(messages)}
        self.engine = Engine()
        self.workspace = TemporaryDirectory()
        self.frontend = Path(self.workspace.name) / "dist"
        (self.frontend / "assets").mkdir(parents=True)
        (self.frontend / "index.html").write_text('<div id="root"></div><script type="module" src="/assets/app-123.js"></script>')
        (self.frontend / "assets/app-123.js").write_bytes(b"export const ready = true;")
        (self.frontend / "assets/style-123.css").write_bytes(b"body{margin:0}")
        (self.frontend / "assets/portrait-123.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (Path(self.workspace.name) / "private.bin").write_bytes(b"not a frontend asset")
        (self.frontend / "assets/linked.bin").symlink_to(Path(self.workspace.name) / "private.bin")
        self.server = ThreadingHTTPServer(("127.0.0.1",0), handler_for(self.engine,self.frontend))
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.workspace.cleanup()

    def request(self, path, payload=None, headers=None):
        request = urllib.request.Request(self.url+path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type":"application/json", **(headers or {})})
        with urllib.request.urlopen(request,timeout=3) as response:
            return response.status, response.read(), response.headers

    def test_built_page_assets_and_status_are_served_locally(self):
        status, html, _ = self.request("/")
        self.assertEqual(status,200)
        self.assertIn(b'id="root"',html)
        _, script, headers = self.request("/assets/app-123.js?v=1")
        self.assertEqual(script,b"export const ready = true;")
        self.assertIn(headers["Content-Type"].split(";")[0],("text/javascript","application/javascript"))
        _, _, headers = self.request("/assets/style-123.css")
        self.assertEqual(headers["Content-Type"],"text/css; charset=utf-8")
        _, portrait, headers = self.request("/assets/portrait-123.png")
        self.assertTrue(portrait.startswith(b"\x89PNG"))
        self.assertEqual(headers["Content-Type"],"image/png")
        self.assertEqual(json.loads(self.request("/api/status")[1])["context_size"],384)

    def test_missing_build_is_actionable_and_api_still_works(self):
        (self.frontend / "index.html").unlink()
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request("/")
        self.assertEqual(error.exception.code,503)
        self.assertIn("frontend run build",json.loads(error.exception.read())["error"])
        self.assertEqual(self.request("/api/status")[0],200)

    def test_unknown_paths_and_traversal_do_not_expose_other_files(self):
        for path in ("/api/missing", "/missing", "/../private.bin",
                     "/assets/%2e%2e/%2e%2e/private.bin", "/assets/linked.bin", "/%00"):
            with self.subTest(path=path), self.assertRaises(urllib.error.HTTPError) as error:
                self.request(path)
            self.assertEqual(error.exception.code,404)

    def test_history_reaches_engine_and_response_reaches_client(self):
        messages = [{"role":"user","content":"My name is Arnav."},
                    {"role":"assistant","content":"Hi, Arnav!"},
                    {"role":"user","content":"What is my name?"}]
        result = json.loads(self.request("/api/chat",{"messages":messages})[1])
        self.assertEqual(self.engine.received,messages)
        self.assertEqual(result["reply"],"Hi, Arnav! What are you making?")

    def test_invalid_roles_bad_json_shape_and_external_origin_are_rejected(self):
        for payload in ({"messages":[{"role":"system","content":"bad role"}]}, []):
            with self.subTest(payload=payload), self.assertRaises(urllib.error.HTTPError) as error:
                self.request("/api/chat",payload)
            self.assertEqual(error.exception.code,400)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request("/api/chat",{"messages":[{"role":"user","content":"hi"}]},
                         headers={"Origin":"https://example.com"})
        self.assertEqual(error.exception.code,403)


if __name__ == "__main__":
    unittest.main()
