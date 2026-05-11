import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request


DEFAULT_BASE_URL = "http://127.0.0.1:9000/v1"
DEFAULT_MODEL = "deepseek-v4-flash-q8"
REMOTE_CLOSE_ERRORS = (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)


def configure_stdio():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Call the local OpenAI-compatible DeepSeek V4 server with long timeouts."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--system", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--repeat-penalty", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--show-request", action="store_true")
    parser.add_argument("--health-first", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("text", nargs="*")
    return parser.parse_args()


def load_prompt(args):
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            return f.read()

    if args.prompt is not None:
        if args.prompt == "-":
            return sys.stdin.read()
        return args.prompt

    if args.text:
        return " ".join(args.text)

    return "你好呀，你是谁呀？"


def build_payload(args, prompt):
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "repeat_penalty": args.repeat_penalty,
        "do_sample": args.do_sample,
        "stream": args.stream,
    }
    if args.max_tokens is not None:
        payload["max_tokens"] = args.max_tokens
    return payload


def post_json(url, payload, timeout):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "text/event-stream" if payload.get("stream") else "application/json",
        },
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=timeout)


def get_json(url, timeout):
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def iter_sse(response):
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            yield "unknown", line
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            yield "done", None
            return
        try:
            yield "json", json.loads(data)
        except json.JSONDecodeError:
            yield "bad_json", data


def extract_stream_piece(payload):
    choices = payload.get("choices") or []
    if not choices:
        return "", None
    choice = choices[0]
    delta = choice.get("delta") or {}
    return delta.get("content") or "", choice.get("finish_reason")


def write_output(path, text):
    if not path:
        return
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def log(message, enabled=True):
    if enabled:
        print(message, file=sys.stderr, flush=True)


def call_stream(url, payload, timeout, verbose):
    start = time.time()
    first_event_at = None
    first_content_at = None
    chunks = 0
    chars = 0
    finish_reason = None
    parts = []

    with post_json(url, payload, timeout) as response:
        first_event_at = time.time()
        log(f"HTTP {response.status}; first_event_sec={first_event_at - start:.3f}", verbose)
        for event_type, event in iter_sse(response):
            now = time.time()
            if event_type == "done":
                print("", flush=True)
                log("STREAM_DONE", verbose)
                return {
                    "ok": True,
                    "text": "".join(parts),
                    "done_sec": now - start,
                    "first_event_sec": (first_event_at or now) - start,
                    "first_content_sec": None if first_content_at is None else first_content_at - start,
                    "chunks": chunks,
                    "chars": chars,
                    "finish_reason": finish_reason,
                }

            if event_type != "json":
                log(f"SSE_{event_type}: {event!r}", verbose)
                continue

            piece, reason = extract_stream_piece(event)
            if reason is not None:
                finish_reason = reason
            if not piece:
                continue

            if first_content_at is None:
                first_content_at = now
                log(f"FIRST_CONTENT_SEC {first_content_at - start:.3f}", verbose)

            chunks += 1
            chars += len(piece)
            parts.append(piece)
            print(piece, end="", flush=True)
            log(f"CHUNK index={chunks} dt={now - start:.3f} chars={len(piece)} total_chars={chars}", verbose)

    return {
        "ok": False,
        "text": "".join(parts),
        "done_sec": time.time() - start,
        "first_event_sec": None if first_event_at is None else first_event_at - start,
        "first_content_sec": None if first_content_at is None else first_content_at - start,
        "chunks": chunks,
        "chars": chars,
        "finish_reason": finish_reason,
        "error": "stream ended without [DONE]",
    }


def call_non_stream(url, payload, timeout):
    start = time.time()
    with post_json(url, payload, timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    done = time.time()
    body = json.loads(raw)
    choices = body.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    text = message.get("content") or ""
    print(text, flush=True)
    return {
        "ok": True,
        "text": text,
        "done_sec": done - start,
        "chars": len(text),
        "finish_reason": choices[0].get("finish_reason") if choices else None,
    }


def main():
    configure_stdio()
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    url = base_url + "/chat/completions"
    prompt = load_prompt(args)
    payload = build_payload(args, prompt)

    log(
        "REQUEST "
        f"url={url} model={args.model} stream={args.stream} "
        f"max_tokens={args.max_tokens} timeout={args.timeout} prompt_chars={len(prompt)}",
        args.verbose,
    )
    if args.show_request:
        log(json.dumps(payload, ensure_ascii=False, indent=2), True)

    try:
        if args.health_first:
            health = get_json(base_url.rsplit("/v1", 1)[0] + "/health", min(args.timeout, 30.0))
            log("HEALTH " + json.dumps(health, ensure_ascii=False), args.verbose)

        result = call_stream(url, payload, args.timeout, args.verbose) if args.stream else call_non_stream(url, payload, args.timeout)
        write_output(args.output_file, result["text"])
        summary = {k: v for k, v in result.items() if k != "text"}
        log("SUMMARY " + json.dumps(summary, ensure_ascii=False), args.verbose)
        return 0 if result.get("ok") else 2
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        log(f"HTTP_ERROR status={exc.code} body={raw}", True)
        return 1
    except (TimeoutError, socket.timeout) as exc:
        log(f"TIMEOUT {type(exc).__name__}: {exc}", True)
        return 3
    except REMOTE_CLOSE_ERRORS as exc:
        log(f"REMOTE_CLOSED {type(exc).__name__}: {exc}", True)
        return 5
    except urllib.error.URLError as exc:
        log(f"URL_ERROR {exc}", True)
        return 4
    except KeyboardInterrupt:
        log("INTERRUPTED", True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
