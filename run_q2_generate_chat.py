import sys
import time

from model_runtime import ModelRuntime


def main():
    runtime = ModelRuntime(default_max_tokens=32)
    prompt = " ".join(sys.argv[1:]) or "请用中文写三句话介绍你自己。"

    status = runtime.status()
    print(f"threads={status['threads']}", flush=True)
    print(f"max_new_tokens={runtime.default_max_tokens}", flush=True)
    print(f"path_exists={status['path_exists']}", flush=True)
    print(f"devices={status['devices']!r}", flush=True)

    print("loading", flush=True)
    t0 = time.time()
    status = runtime.load()
    print(f"loaded_sec={status.get('loaded_sec', time.time() - t0):.1f}", flush=True)
    print(f"model_type={status['model_type']}", flush=True)

    query = [{"role": "user", "content": prompt}]
    print("generating", flush=True)
    t1 = time.time()
    parts = []
    for piece in runtime.stream_chat(
        query,
        max_tokens=runtime.default_max_tokens,
        do_sample=False,
        top_k=1,
        temperature=1.0,
        repeat_penalty=1.0,
    ):
        parts.append(piece)
        print(piece, end="", flush=True)
    print("", flush=True)
    output = "".join(parts)
    print(f"done_sec={time.time() - t1:.1f}", flush=True)
    print(f"output={output!r}", flush=True)


if __name__ == "__main__":
    main()

