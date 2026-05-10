import os
import sys
import time


ROOT = os.path.dirname(os.path.abspath(__file__))
CUDA_BIN = os.path.join(ROOT, "cuda-12.4-toolkit", "bin")
TOOLS_DIR = os.path.join(ROOT, "fastllm-master", "build-cuda-ninja", "tools")

os.environ["PATH"] = CUDA_BIN + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, TOOLS_DIR)

os.environ.setdefault("FASTLLM_DSV4_DISABLE_PREFIX_CACHE", "1")
os.environ.setdefault("FASTLLM_PROFILE_DEEPSEEKV4", "1")
os.environ.setdefault("FASTLLM_DEBUG_RESPONSE_LOOP", "1")
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


def main():
    threads = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    path = model_path()
    print(f"threads={threads}", flush=True)
    print(f"path_exists={os.path.exists(path)}", flush=True)
    print(
        "devices="
        + repr(
            {
                "cpu": llm.has_device("cpu"),
                "cuda": llm.has_device("cuda"),
                "disk": llm.has_device("disk"),
            }
        ),
        flush=True,
    )

    llm.set_cpu_threads(threads)
    llm.set_cpu_low_mem(True)
    llm.set_cuda_embedding(False)
    llm.set_gpu_mem_ratio(0.97)
    llm.set_device_map("cuda")
    llm.set_device_map("disk", True)

    t0 = time.time()
    model = llm.model(path, dtype="float16", kv_cache_dtype="fp8_e4m3")
    print(f"loaded_sec={time.time() - t0:.1f}", flush=True)
    print(f"model_type={model.get_type()}", flush=True)

    model.direct_query = True
    model.enable_thinking = False
    model.set_save_history(False)
    model.set_verbose(1)
    model.set_atype("float32")
    model.set_moe_atype("float32")
    model.set_moe_experts(1)
    model.set_kv_cache_limit("2g")
    model.set_chunked_prefill_size(1)

    tokens = model.tokenizer_encode_string("hi") or [2]
    tokens = tokens[:1]
    print(f"input_tokens={tokens}", flush=True)
    t1 = time.time()
    pieces = []
    for piece in model.stream_response_raw(
        tokens,
        max_length=1,
        do_sample=False,
        top_k=1,
        temperature=1.0,
        repeat_penalty=1.0,
        one_by_one=True,
        stop_token_ids=[1],
    ):
        pieces.append(piece)
        print(f"piece_bytes={piece!r}", flush=True)
    print(f"done_sec={time.time() - t1:.1f}", flush=True)
    print(f"pieces={pieces!r}", flush=True)


if __name__ == "__main__":
    main()
