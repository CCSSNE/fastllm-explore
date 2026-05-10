import gc
import os
import sys
import threading
import time


for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


ROOT = os.path.dirname(os.path.abspath(__file__))
CUDA_BIN = os.path.join(ROOT, "cuda-12.4-toolkit", "bin")
TOOLS_DIR_CLEAN = os.path.join(ROOT, "fastllm-master", "build-cuda-ninja-clean", "tools")
TOOLS_DIR_LEGACY = os.path.join(ROOT, "fastllm-master", "build-cuda-ninja", "tools")
TOOLS_DIR = TOOLS_DIR_CLEAN if os.path.exists(os.path.join(TOOLS_DIR_CLEAN, "ftllm", "fastllm_tools.dll")) else TOOLS_DIR_LEGACY

os.environ["PATH"] = CUDA_BIN + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("FASTLLM_DSV4_DISABLE_PREFIX_CACHE", "1")
os.environ.setdefault("FASTLLM_DSV4_DISABLE_CUDA_HCPRE", "1")
os.environ.setdefault("FASTLLM_DSV4_DISABLE_CUDA_WOA_HCPOST", "1")
os.environ.setdefault("FASTLLM_DISK_MOE_LOAD_THREADS", "16")
os.environ.setdefault("USE_OLD_ENGINE", "1")

if os.name == "nt":
    for dll_dir in (CUDA_BIN, TOOLS_DIR):
        try:
            os.add_dll_directory(dll_dir)
        except Exception:
            pass

sys.path.insert(0, TOOLS_DIR)

from ftllm import llm


def default_model_path():
    local_path = os.path.join(ROOT, "cyberneurova-DeepSeek-V4-Flash-abliterated-Q8_0.gguf")
    if os.path.exists(local_path):
        return local_path

    sep = chr(92)
    return (
        "F:"
        + sep
        + chr(0x4E0B)
        + chr(0x8F7D)
        + sep
        + "cyberneurova-DeepSeek-V4-Flash-abliterated-Q8_0.gguf"
    )


class ModelNotLoadedError(RuntimeError):
    pass


class ModelRuntime:
    def __init__(
        self,
        model_path=None,
        model_name=None,
        threads=None,
        kv_cache_limit=None,
        default_max_tokens=None,
        chunked_prefill_size=None,
    ):
        self.model_path = model_path or os.environ.get("DSV4_MODEL_PATH") or default_model_path()
        self.model_name = model_name or os.environ.get("OPENAI_MODEL_NAME", "deepseek-v4-flash-q8")
        self.threads = int(threads or os.environ.get("DSV4_THREADS") or os.environ.get("Q2_THREADS", "16"))
        self.kv_cache_limit = kv_cache_limit or os.environ.get("DSV4_KV_CACHE_LIMIT") or os.environ.get("Q2_KV_CACHE_LIMIT", "8g")
        self.default_max_tokens = int(
            default_max_tokens
            or os.environ.get("DSV4_MAX_NEW_TOKENS")
            or os.environ.get("Q2_MAX_NEW_TOKENS", "128")
        )
        self.chunked_prefill_size = int(
            chunked_prefill_size
            or os.environ.get("DSV4_CHUNKED_PREFILL_SIZE")
            or os.environ.get("Q2_CHUNKED_PREFILL_SIZE", "8")
        )
        self.model = None
        self.model_type = None
        self.loaded_at = None
        self._lock = threading.RLock()

    def status(self):
        return {
            "loaded": self.model is not None,
            "model": self.model_name,
            "model_type": self.model_type,
            "model_path": self.model_path,
            "path_exists": os.path.exists(self.model_path),
            "threads": self.threads,
            "kv_cache_limit": self.kv_cache_limit,
            "default_max_tokens": self.default_max_tokens,
            "chunked_prefill_size": self.chunked_prefill_size,
            "tools_dir": TOOLS_DIR,
            "loaded_at": self.loaded_at,
            "devices": {
                "cpu": llm.has_device("cpu"),
                "cuda": llm.has_device("cuda"),
                "disk": llm.has_device("disk"),
            },
        }

    def load(self):
        with self._lock:
            if self.model is not None:
                return self.status()

            llm.set_cpu_threads(self.threads)
            llm.set_cpu_low_mem(True)
            llm.set_cuda_embedding(False)
            llm.set_gpu_mem_ratio(0.99)
            llm.set_device_map("cuda")
            llm.set_device_map("disk", True)

            t0 = time.time()
            model = llm.model(self.model_path, dtype="float16", kv_cache_dtype="fp8_e4m3")
            self.loaded_at = int(time.time())
            self.model_type = model.get_type()

            model.direct_query = False
            model.enable_thinking = False
            model.set_save_history(False)
            model.set_verbose(1)
            model.set_atype("float32")
            model.set_moe_atype("float32")
            model.set_kv_cache_limit(self.kv_cache_limit)
            model.set_chunked_prefill_size(self.chunked_prefill_size)

            self.model = model
            status = self.status()
            status["loaded_sec"] = round(time.time() - t0, 1)
            return status

    def unload(self):
        with self._lock:
            old_model = self.model
            self.model = None
            self.model_type = None
            self.loaded_at = None
            if old_model is not None:
                del old_model
            gc.collect()
            return self.status()

    def stream_chat(
        self,
        messages,
        max_tokens=None,
        do_sample=False,
        top_k=1,
        temperature=1.0,
        repeat_penalty=1.0,
    ):
        with self._lock:
            if self.model is None:
                raise ModelNotLoadedError("model is not loaded")

            normalized = normalize_messages(messages)
            limit = int(max_tokens or self.default_max_tokens)
            for piece in self.model.stream_response(
                normalized,
                max_length=limit,
                do_sample=bool(do_sample),
                top_k=int(top_k),
                temperature=float(temperature),
                repeat_penalty=float(repeat_penalty),
                one_by_one=True,
                stop_token_ids=None,
            ):
                yield piece

    def complete_chat(self, messages, **kwargs):
        return "".join(self.stream_chat(messages, **kwargs))


def normalize_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")

        role = str(message.get("role") or "user")
        content = normalize_content(message.get("content", ""))
        normalized.append({"role": role, "content": content})

    return normalized


def normalize_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)
