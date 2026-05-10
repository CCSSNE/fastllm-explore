import os
import sys
import time


ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "fastllm-master", "build-cuda-ninja", "tools"))

from ftllm import llm
from ftllm.encoding_dsv4 import encode_messages


def model_path() -> str:
    sep = chr(92)
    return (
        "D:"
        + sep
        + chr(0x4E0B)
        + chr(0x8F7D)
        + sep
        + "cyberneurova-DeepSeek-V4-Flash-abliterated-Q2_K.gguf"
    )


def log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    path = model_path()
    os.environ.setdefault("FASTLLM_DSV4_DISABLE_PREFIX_CACHE", "1")

    log(f"path_exists={os.path.exists(path)}")
    log(f"path_utf8={path.encode('utf-8')!r}")
    log(
        "devices="
        + repr(
            {
                "cpu": llm.has_device("cpu"),
                "cuda": llm.has_device("cuda"),
                "disk": llm.has_device("disk"),
            }
        )
    )

    llm.set_cpu_threads(2)
    llm.set_cpu_low_mem(True)
    llm.set_gpu_mem_ratio(0.96)
    llm.set_device_map("cuda")
    llm.set_device_map("disk", True)

    log("loading")
    t0 = time.time()
    model = llm.model(path, dtype="float16", kv_cache_dtype="fp8_e4m3")
    log(f"loaded_sec={time.time() - t0:.1f}")
    log(f"model_type={model.get_type()}")
    log(f"max_input={model.get_max_input_len()}")

    model.direct_query = True
    model.enable_thinking = False
    model.set_atype("float32")
    model.set_moe_experts(1)
    model.set_kv_cache_limit("256m")
    model.set_chunked_prefill_size(32)

    prompt = encode_messages(
        [{"role": "user", "content": "用中文说一句你好。"}], thinking_mode="chat"
    )
    log(f"prompt_len={len(prompt)}")
    log("generating")

    t1 = time.time()
    out = []
    for piece in model.stream_response(
        prompt,
        max_length=8,
        do_sample=False,
        top_k=1,
        temperature=1.0,
    ):
        if piece:
            out.append(piece)
            print(piece, end="", flush=True)

    log("")
    log(f"done_sec={time.time() - t1:.1f}")
    log(f"output={''.join(out)!r}")


if __name__ == "__main__":
    main()
