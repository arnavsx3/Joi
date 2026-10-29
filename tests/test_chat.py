from http.server import ThreadingHTTPServer
import json
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

import torch

from joi.model import Joi, ModelConfig
from joi.serve import handler_for


class StopSequenceTests(unittest.TestCase):
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
        self.server = ThreadingHTTPServer(("127.0.0.1",0), handler_for(self.engine))
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, path, payload=None, headers=None):
        request = urllib.request.Request(self.url+path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type":"application/json", **(headers or {})})
        with urllib.request.urlopen(request,timeout=3) as response:
            return response.status, response.read(), response.headers

    def test_page_portrait_and_status_are_served_locally(self):
        status, html, _ = self.request("/")
        self.assertEqual(status,200)
        self.assertIn(b"Come as you are.",html)
        _, portrait, headers = self.request("/assets/joi-portrait.png")
        self.assertTrue(portrait.startswith(b"\x89PNG"))
        self.assertEqual(headers["Content-Type"],"image/png")
        self.assertEqual(json.loads(self.request("/api/status")[1])["context_size"],384)

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
