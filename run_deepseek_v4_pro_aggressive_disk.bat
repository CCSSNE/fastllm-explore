@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "DSV4_MODEL_PATH=F:\model\DeepSeek-V4-Pro-GGUF-Q4_K_M"
set "OPENAI_MODEL_NAME=deepseek-v4-pro-q4-k-m"
set "DSV4_THREADS=8"
set "DSV4_KV_CACHE_LIMIT=1g"
set "DSV4_CHUNKED_PREFILL_SIZE=2"
set "DSV4_MAX_NEW_TOKENS=16"
set "DSV4_GPU_MEM_RATIO=0.85"
set "DSV4_DEVICE_MAP=cuda"
set "DSV4_MOE_DEVICE_MAP=disk"
set "FASTLLM_DISK_MOE_LOAD_THREADS=8"
set "FASTLLM_DISK_MOE_CACHE_MB=4096"
python "%~dp0openai_compatible_server.py" --no-auto-load %*
