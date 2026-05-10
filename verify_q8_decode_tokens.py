import os
import sys
import time

from model_runtime import ModelRuntime


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    max_tokens = int(os.environ.get("DSV4_MAX_NEW_TOKENS", "6"))
    prompt = " ".join(sys.argv[1:]) or "请用中文连续写一段不少于五十字的话，主题是本地模型推理速度测试。"

    runtime = ModelRuntime(default_max_tokens=max_tokens)
    print(f"STATUS {runtime.status()!r}", flush=True)

    t0 = time.time()
    status = runtime.load()
    print(f"loaded_sec={time.time() - t0:.1f}", flush=True)
    print(f"STATUS {status!r}", flush=True)

    query = [{"role": "user", "content": prompt}]
    parts = []
    gen_start = time.time()
    last = gen_start

    for index, piece in enumerate(
        runtime.stream_chat(
            query,
            max_tokens=max_tokens,
            do_sample=False,
            top_k=1,
            temperature=1.0,
            repeat_penalty=1.0,
        ),
        start=1,
    ):
        now = time.time()
        parts.append(piece)
        print(
            f"TOKEN_EVENT index={index} dt={now - last:.3f} total={now - gen_start:.3f} piece={piece!r}",
            flush=True,
        )
        last = now

    output = "".join(parts)
    print(f"done_sec={time.time() - gen_start:.1f}", flush=True)
    print(f"output={output!r}", flush=True)


if __name__ == "__main__":
    main()
