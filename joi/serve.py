"""Serve Joi's local chat screen and inference API on this laptop."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse

from joi.chat import ChatEngine

ROOT = Path(__file__).resolve().parent.parent
FILES = {"/": ("web/index.html", "text/html; charset=utf-8"),
         "/app.css": ("web/app.css", "text/css; charset=utf-8"),
         "/app.js": ("web/app.js", "application/javascript; charset=utf-8"),
         "/assets/joi-portrait.png": ("assets/joi-portrait.png", "image/png")}


def handler_for(engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, status, data, content_type="application/json; charset=utf-8"):
            body = json.dumps(data).encode() if isinstance(data, dict) else data
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/status":
                self.respond(200, engine.status())
            elif path in FILES:
                filename, content_type = FILES[path]
                self.respond(200, (ROOT / filename).read_bytes(), content_type)
            else:
                self.respond(404, {"error":"Not found"})

        def do_POST(self):
            if self.path != "/api/chat":
                self.respond(404, {"error":"Not found"})
                return
            origin = self.headers.get("Origin")
            port = self.server.server_address[1]
            if origin and origin not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
                self.respond(403, {"error":"Use Joi's local chat page"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Please send a smaller conversation")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("Send a conversation object")
                messages = request.get("messages")
                if not isinstance(messages, list) or not 1 <= len(messages) <= 40:
                    raise ValueError("A conversation needs between 1 and 40 messages")
                for index, message in enumerate(messages):
                    expected = "user" if index % 2 == 0 else "assistant"
                    if not isinstance(message, dict) or message.get("role") != expected:
                        raise ValueError("Conversation turns must alternate between you and Joi")
                    content = message.get("content")
                    if not isinstance(content, str) or len(content) > 4000:
                        raise ValueError("Please send shorter text messages")
                if messages[-1]["role"] != "user":
                    raise ValueError("Send your message before asking Joi to reply")
                self.respond(200, engine.reply(messages, seed=42 + len(messages)))
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(400, {"error":str(exc)})
            except Exception:
                self.respond(500, {"error":"Joi couldn't finish that reply. Please try again."})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/dialogue-cpu/best.pt")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Choose a port between 1 and 65535")
    try:
        engine = ChatEngine(args.checkpoint)
        server = ThreadingHTTPServer(("127.0.0.1",args.port), handler_for(engine))
    except (ValueError, FileNotFoundError, OSError) as exc:
        parser.error(str(exc))
    print(f"Joi is ready at http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
