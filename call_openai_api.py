import json
import urllib.request


url = "http://127.0.0.1:9000/v1/chat/completions"

payload = {
    "model": "deepseek-v4-flash-q8",
    "messages": [
        {"role": "user", "content": "写一篇1000字小说"},
    ],
    "stream": True,
}

request = urllib.request.Request(
    url,
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json; charset=utf-8"},
    method="POST",
)

with urllib.request.urlopen(request) as response:
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data: "):
            continue

        data = line[6:]
        if data == "[DONE]":
            break

        chunk = json.loads(data)
        if "error" in chunk:
            print(chunk["error"].get("message", chunk["error"]))
            break

        choices = chunk.get("choices")
        if not choices:
            continue

        content = choices[0]["delta"].get("content")
        if content:
            print(content, end="", flush=True)

print()
