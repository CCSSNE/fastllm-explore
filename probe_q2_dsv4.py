import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
CUDA_BIN = os.path.join(ROOT, "cuda-12.4-toolkit", "bin")
TOOLS_DIR = os.path.join(ROOT, "fastllm-master", "build-cuda-ninja", "tools")

os.environ["PATH"] = CUDA_BIN + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, TOOLS_DIR)

os.environ.setdefault("FASTLLM_DSV4_DISABLE_PREFIX_CACHE", "1")
os.environ.setdefault("FASTLLM_DSV4_DISABLE_CUDA_HCPRE", "1")
os.environ.setdefault("FASTLLM_DSV4_DISABLE_CUDA_WOA_HCPOST", "1")
os.environ.setdefault("USE_OLD_ENGINE", "1")

from ftllm import llm


def model_path():
    sep = chr(92)
    return (
        "D:"
        + sep
        + chr(0x4E0B)
        + chr(0x8F7D)
        + sep
        + "cyberneurova-DeepSeek-V4-Flash-abliterated-Q2_K.gguf"
    )


def decode_bytes(parts):
    data = b"".join(parts)
    return data.decode("utf-8", errors="replace")


def main():
    prompt = " ".join(sys.argv[1:]) or "你好，请回答：一加一等于几？请只用一句中文回答。"
    threads = int(os.environ.get("Q2_THREADS", "16"))
    print(f"threads={threads}", flush=True)
    llm.set_cpu_threads(threads)
    llm.set_cpu_low_mem(True)
    llm.set_cuda_embedding(False)
    llm.set_gpu_mem_ratio(0.99)
    llm.set_device_map("cuda")
    llm.set_device_map("disk", True)

    t0 = time.time()
    model = llm.model(model_path(), dtype="float16", kv_cache_dtype="fp8_e4m3")
    print(f"loaded_sec={time.time() - t0:.1f}", flush=True)
    print(f"struct={model.get_struct()}", flush=True)
    print(f"type={model.get_type()}", flush=True)
    print(f"is_dsv4={model._is_deepseek_v4()}", flush=True)

    model.direct_query = os.environ.get("Q2_DIRECT_QUERY", "0") == "1"
    model.enable_thinking = os.environ.get("Q2_ENABLE_THINKING", "0") == "1"
    model.set_save_history(False)
    model.set_verbose(0)
    model.set_atype("float32")
    model.set_moe_atype("float32")
    model.set_kv_cache_limit("4g")
    model.set_chunked_prefill_size(1)

    conversation = [{"role": "user", "content": prompt}]
    rendered = model.get_prompt(prompt, [])
    print(f"direct_query={model.direct_query}", flush=True)
    print(f"enable_thinking={model.enable_thinking}", flush=True)
    print("prompt_repr=" + repr(rendered), flush=True)
    tokens = model.tokenizer_encode_string(rendered)
    print(f"input_token_count={len(tokens)}", flush=True)
    print("input_tokens_head=" + repr(tokens[:16]), flush=True)
    print("input_tokens_tail=" + repr(tokens[-16:]), flush=True)

    ns = [int(x) for x in os.environ.get("Q2_PROBE_NS", "1,2,4,8,16").split(",") if x.strip()]
    for n in ns:
        t1 = time.time()
        parts = list(
            model.stream_response_raw(
                tokens,
                max_length=n,
                do_sample=False,
                top_k=1,
                temperature=1.0,
                repeat_penalty=1.0,
                one_by_one=True,
                stop_token_ids=None,
            )
        )
        print(f"n={n} sec={time.time() - t1:.1f} bytes={parts!r}", flush=True)
        print(f"n={n} text={decode_bytes(parts)!r}", flush=True)


if __name__ == "__main__":
    main()
