"""Serve Joi's local chat screen and inference API on this laptop."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlparse

from joi.chat import CompanionEngine

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend/dist"


def handler_for(engine, frontend_root=FRONTEND):
    frontend_root = Path(frontend_root).resolve()
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
            path = unquote(urlparse(self.path).path)
            if path == "/api/status":
                self.respond(200, engine.status())
                return
            if path.startswith("/api/"):
                self.respond(404, {"error":"Not found"})
                return
            relative = path.lstrip("/") or "index.html"
            if relative == "index.html" and not (frontend_root / relative).is_file():
                self.respond(503, {"error":"Build Joi's frontend first: npm --prefix frontend ci, then npm --prefix frontend run build."})
                return
            try:
                filename = (frontend_root / relative).resolve()
                if not filename.is_relative_to(frontend_root) or not filename.is_file():
                    raise FileNotFoundError
                body = filename.read_bytes()
            except (OSError, ValueError):
                self.respond(404, {"error":"Not found"})
                return
            content_type = mimetypes.guess_type(filename.name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type == "application/javascript":
                content_type += "; charset=utf-8"
            self.respond(200, body, content_type)

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
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/joi-chat/best.pt")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-recall-specialist",action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Choose a port between 1 and 65535")
    try:
        engine = CompanionEngine(args.checkpoint,use_recall=not args.no_recall_specialist)
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
