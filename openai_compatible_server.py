import argparse
import json
import time
import uuid
from http import HTTPStatus
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

from model_runtime import ModelNotLoadedError, ModelRuntime


RUNTIME = None
CLIENT_DISCONNECT_ERRORS = (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)


class OpenAICompatibleHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "DeepSeekV4Local/0.1"

    def do_GET(self):
        if self.path == "/health":
            self.send_json(HTTPStatus.OK, RUNTIME.status())
            return

        if self.path == "/v1/models":
            now = int(time.time())
            self.send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": RUNTIME.model_name,
                            "object": "model",
                            "created": now,
                            "owned_by": "local",
                        }
                    ],
                },
            )
            return

        self.send_error_json(HTTPStatus.NOT_FOUND, "not_found", "Not found")

    def do_POST(self):
        if self.path == "/admin/load":
            self.send_json(HTTPStatus.OK, RUNTIME.load())
            return

        if self.path == "/admin/unload":
            self.send_json(HTTPStatus.OK, RUNTIME.unload())
            return

        if self.path == "/v1/chat/completions":
            self.handle_chat_completions()
            return

        self.send_error_json(HTTPStatus.NOT_FOUND, "not_found", "Not found")

    def handle_chat_completions(self):
        try:
            body = self.read_json()
            messages = body.get("messages")
            max_tokens = body.get("max_tokens")
            stream = bool(body.get("stream", False))
            temperature = body.get("temperature", 1.0)
            top_k = body.get("top_k", 1)
            repeat_penalty = body.get("repeat_penalty", 1.0)
            do_sample = bool(body.get("do_sample", False))

            if stream:
                self.stream_chat_response(
                    messages,
                    max_tokens=max_tokens,
                    do_sample=do_sample,
                    top_k=top_k,
                    temperature=temperature,
                    repeat_penalty=repeat_penalty,
                )
                return

            text = RUNTIME.complete_chat(
                messages,
                max_tokens=max_tokens,
                do_sample=do_sample,
                top_k=top_k,
                temperature=temperature,
                repeat_penalty=repeat_penalty,
            )
            self.send_json(HTTPStatus.OK, chat_completion_response(text))
        except ModelNotLoadedError as exc:
            self.send_error_json(HTTPStatus.CONFLICT, "model_not_loaded", str(exc))
        except ValueError as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "invalid_request_error", str(exc))
        except CLIENT_DISCONNECT_ERRORS:
            print("client disconnected during response", flush=True)
        except Exception as exc:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, "server_error", str(exc))

    def stream_chat_response(self, messages, **kwargs):
        completion_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        first = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": RUNTIME.model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
        try:
            self.write_sse(first)

            generator = RUNTIME.stream_chat(messages, **kwargs)
            for piece in generator:
                if not piece:
                    continue
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": RUNTIME.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": piece},
                            "finish_reason": None,
                        }
                    ],
                }
                self.write_sse(chunk)

            final = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": RUNTIME.model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            self.write_sse(final)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except CLIENT_DISCONNECT_ERRORS:
            if "generator" in locals():
                generator.close()
            print("client disconnected during stream", flush=True)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def send_json(self, status, body):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_error_json(self, status, error_type, message):
        self.send_json(
            status,
            {
                "error": {
                    "message": message,
                    "type": error_type,
                    "param": None,
                    "code": None,
                }
            },
        )

    def write_sse(self, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.wfile.write(b"data: " + data + b"\n\n")
        self.wfile.flush()

    def log_message(self, fmt, *args):
        print(
            "%s - - [%s] %s"
            % (self.client_address[0], self.log_date_time_string(), fmt % args),
            flush=True,
        )


def chat_completion_response(text):
    now = int(time.time())
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": now,
        "model": RUNTIME.model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="OpenAI compatible server for local DeepSeek V4 GGUF")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--no-auto-load", action="store_true")
    return parser.parse_args()


def main():
    global RUNTIME
    args = parse_args()
    RUNTIME = ModelRuntime()

    if not args.no_auto_load:
        print("loading model...", flush=True)
        status = RUNTIME.load()
        print(json.dumps(status, ensure_ascii=False), flush=True)

    server = ThreadingHTTPServer((args.host, args.port), OpenAICompatibleHandler)
    print(f"listening on http://{args.host}:{args.port}", flush=True)
    print(f"openai base url: http://{args.host}:{args.port}/v1", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutting down", flush=True)
    finally:
        RUNTIME.unload()
        server.server_close()


if __name__ == "__main__":
    main()
