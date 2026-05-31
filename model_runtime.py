import gc
import os
import re
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
os.environ.setdefault("FASTLLM_DISK_MOE_LOAD_THREADS", "12")
os.environ.setdefault("FASTLLM_DISK_MOE_CACHE_MB", "8192")
os.environ.setdefault("USE_OLD_ENGINE", "1")

if os.name == "nt":
    for dll_dir in (CUDA_BIN, TOOLS_DIR):
        try:
            os.add_dll_directory(dll_dir)
        except Exception:
            pass

sys.path.insert(0, TOOLS_DIR)

from ftllm import llm


GGUF_SHARD_RE = re.compile(r"^(.+)-(\d+)-of-(\d+)\.gguf$", re.IGNORECASE)


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


def resolve_model_path(model_path):
    if os.path.isdir(model_path):
        return resolve_gguf_directory(model_path)
    return normalize_gguf_shard_path(model_path)


def resolve_gguf_directory(model_dir):
    names = sorted(name for name in os.listdir(model_dir) if name.lower().endswith(".gguf"))
    if len(names) == 1:
        return os.path.join(model_dir, names[0])

    first_shards = []
    for name in names:
        match = GGUF_SHARD_RE.match(name)
        if match and int(match.group(2)) == 1:
            first_shards.append(name)

    if len(first_shards) == 1:
        return os.path.join(model_dir, first_shards[0])
    if len(first_shards) > 1:
        raise ValueError("model directory has multiple GGUF shard entrypoints: " + ", ".join(first_shards))

    raise ValueError("model directory must contain one .gguf file or one *-00001-of-*.gguf first shard")


def normalize_gguf_shard_path(model_path):
    directory, name = os.path.split(model_path)
    match = GGUF_SHARD_RE.match(name)
    if not match:
        return model_path

    base_name = match.group(1)
    digits = len(match.group(2))
    total = int(match.group(3))
    first_name = f"{base_name}-{1:0{digits}d}-of-{total:0{digits}d}.gguf"
    return os.path.join(directory, first_name)


def describe_model_files(input_path, resolved_path):
    info = {
        "input_path": input_path,
        "resolved_path": resolved_path,
        "input_is_directory": os.path.isdir(input_path),
        "resolved_exists": os.path.exists(resolved_path),
        "is_sharded": False,
    }
    if not os.path.exists(resolved_path):
        return info

    directory, name = os.path.split(resolved_path)
    match = GGUF_SHARD_RE.match(name)
    if not match:
        size = os.path.getsize(resolved_path)
        info.update(
            {
                "file_count": 1,
                "total_bytes": size,
                "total_gib": bytes_to_gib(size),
            }
        )
        return info

    base_name = match.group(1)
    digits = len(match.group(2))
    total = int(match.group(3))
    expected_names = [
        f"{base_name}-{i:0{digits}d}-of-{total:0{digits}d}.gguf"
        for i in range(1, total + 1)
    ]
    existing_names = [item for item in expected_names if os.path.exists(os.path.join(directory, item))]
    total_bytes = sum(os.path.getsize(os.path.join(directory, item)) for item in existing_names)
    missing_names = [item for item in expected_names if item not in set(existing_names)]

    info.update(
        {
            "is_sharded": True,
            "file_count": len(existing_names),
            "expected_file_count": total,
            "missing_files": missing_names,
            "first_file": expected_names[0],
            "last_file": expected_names[-1],
            "total_bytes": total_bytes,
            "total_gib": bytes_to_gib(total_bytes),
        }
    )
    return info


def bytes_to_gib(size):
    return round(size / (1024 ** 3), 2)


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
        self.model_path_input = model_path or os.environ.get("DSV4_MODEL_PATH") or default_model_path()
        self.model_path = resolve_model_path(self.model_path_input)
        self.model_name = model_name or os.environ.get("OPENAI_MODEL_NAME", "deepseek-v4-flash-q8")
        self.threads = int(threads or os.environ.get("DSV4_THREADS") or os.environ.get("Q2_THREADS", "12"))
        self.kv_cache_limit = kv_cache_limit or os.environ.get("DSV4_KV_CACHE_LIMIT") or os.environ.get("Q2_KV_CACHE_LIMIT", "8g")
        self.default_max_tokens = int(
            default_max_tokens
            or os.environ.get("DSV4_MAX_NEW_TOKENS")
            or os.environ.get("Q2_MAX_NEW_TOKENS", "1000000")
        )
        self.chunked_prefill_size = int(
            chunked_prefill_size
            or os.environ.get("DSV4_CHUNKED_PREFILL_SIZE")
            or os.environ.get("Q2_CHUNKED_PREFILL_SIZE", "8")
        )
        self.gpu_mem_ratio = float(os.environ.get("DSV4_GPU_MEM_RATIO", "0.99"))
        self.device_map = os.environ.get("DSV4_DEVICE_MAP", "cuda")
        self.moe_device_map = os.environ.get("DSV4_MOE_DEVICE_MAP", "disk")
        self.model = None
        self.model_type = None
        self.loaded_at = None
        self.loading = False
        self.last_error = None
        self._lock = threading.RLock()

    def status(self):
        return {
            "loaded": self.model is not None,
            "loading": self.loading,
            "last_error": self.last_error,
            "model": self.model_name,
            "model_path_input": self.model_path_input,
            "model_type": self.model_type,
            "model_path": self.model_path,
            "path_exists": os.path.exists(self.model_path),
            "model_files": describe_model_files(self.model_path_input, self.model_path),
            "threads": self.threads,
            "kv_cache_limit": self.kv_cache_limit,
            "default_max_tokens": self.default_max_tokens,
            "chunked_prefill_size": self.chunked_prefill_size,
            "gpu_mem_ratio": self.gpu_mem_ratio,
            "device_map": self.device_map,
            "moe_device_map": self.moe_device_map,
            "disk_moe_cache_mb": os.environ.get("FASTLLM_DISK_MOE_CACHE_MB"),
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

            self.loading = True
            self.last_error = None
            t0 = time.time()
            try:
                llm.set_cpu_threads(self.threads)
                llm.set_cpu_low_mem(True)
                llm.set_cuda_embedding(False)
                llm.set_max_tokens(self.default_max_tokens)
                llm.set_gpu_mem_ratio(self.gpu_mem_ratio)
                llm.set_device_map(self.device_map)
                llm.set_device_map(self.moe_device_map, True)

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
                self.loading = False
                status = self.status()
                status["loaded_sec"] = round(time.time() - t0, 1)
                return status
            except Exception as exc:
                self.last_error = str(exc)
                raise
            finally:
                self.loading = False

    def unload(self):
        with self._lock:
            old_model = self.model
            self.model = None
            self.model_type = None
            self.loaded_at = None
            self.loading = False
            if old_model is not None:
                del old_model
            gc.collect()
            return self.status()

    def ensure_loaded(self):
        with self._lock:
            if self.model is None:
                raise ModelNotLoadedError("model is not loaded")

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
            limit = self.default_max_tokens if max_tokens is None else int(max_tokens)
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
