# OpenAI Server 使用说明

## 方式 A：一条命令启动并加载模型

在 PowerShell 里进入项目目录：

```powershell
cd D:\AI\LLM\deepseek-v4部署
```

启动 OpenAI 兼容 server：

```powershell
python .\openai_compatible_server.py
```

这条命令会自动做两件事：

1. 加载模型：

```text
D:\AI\LLM\deepseek-v4部署\cyberneurova-DeepSeek-V4-Flash-abliterated-Q8_0.gguf
```

2. 启动本地 OpenAI 兼容 API：

```text
http://127.0.0.1:9000/v1
```

看到类似下面的输出，就说明 server 已经启动：

```text
listening on http://127.0.0.1:9000
openai base url: http://127.0.0.1:9000/v1
```

## 客户端填写

在支持 OpenAI API 的客户端里这样填：

```text
Base URL: http://127.0.0.1:9000/v1
API Key: local
Model: deepseek-v4-flash-q8
```

## DeepSeek V4 Pro Q4_K_M

推荐使用已经验证能生成 token 的激进硬盘模式。这个入口只启动 API，不会自动加载 21 分片大模型：

```powershell
.\run_deepseek_v4_pro_aggressive_disk.bat
```

先检查分片状态：

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:9000/health -Method Get
```

确认 `model_path` 指向：

```text
F:\model\DeepSeek-V4-Pro-GGUF-Q4_K_M\deepseek-ai-DeepSeek-V4-Pro-Q4_K_M-00001-of-00021.gguf
```

并确认 `model_files.missing_files` 为空后，再手动加载：

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:9000/admin/load -Method Post
```

客户端模型名：

```text
deepseek-v4-pro-q4-k-m
```

已验证结果：

```text
loaded_sec: 367.3 秒
model_type: deepseek_v4
输出 token 1: 能
输出 token 2: 跑
最终输出: 能跑
```

本次跑通参数：

```text
DSV4_THREADS=8
DSV4_KV_CACHE_LIMIT=1g
DSV4_CHUNKED_PREFILL_SIZE=2
DSV4_GPU_MEM_RATIO=0.85
DSV4_DEVICE_MAP=cuda
DSV4_MOE_DEVICE_MAP=disk
FASTLLM_DISK_MOE_CACHE_MB=4096
FASTLLM_DISK_MOE_LOAD_THREADS=8
```

这组参数的取舍是：主设备用 CUDA，MoE 权重走硬盘，硬盘 MoE 缓存给到 4GB，显存比例给到 0.85。它会明显吃满显存和内存，但可以完成加载、prefill 和 token 生成。更省资源的极限硬盘模式曾长时间卡在加载 0%；更激进的近乎全量占用配置会让系统严重换页。当前配置是为了“能跑起来并生成 token”的中间点。

如果只是诊断路径和分片状态，也可以使用较保守入口：

```powershell
.\run_deepseek_v4_pro_server.bat
```

## 检查状态

另开一个 PowerShell，执行：

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:9000/health -Method Get
```

如果返回里有：

```text
loaded : True
```

就说明模型已经加载。

## 停止服务

回到运行 server 的 PowerShell 窗口，按：

```text
Ctrl + C
```

server 关闭后，模型也会一起卸载。
