# fastllm Explore

用硬盘来推理 DeepSeek V4。

本仓库提供一个本地 OpenAI 兼容 server，用于加载 DeepSeek V4 Flash GGUF 模型并暴露 `/v1` API。

## 说明

这套代码已经在我的电脑上跑通了，我已经完成了模型侧和 server 侧的大部分适配。换到你的电脑上时，通常还需要按本机环境调整模型输入路径、CUDA、GPU 驱动和 fastllm 相关依赖。

建议直接把本仓库交给 Codex，让它根据你的显卡、CUDA 版本和模型文件位置做本地适配。我只保证这套配置在我的电脑上可用；剩下的本机路径和硬件环境，需要在你的机器上重新对齐。

## 硬件与效果

测试模型：

- DeepSeek V4 Flash abliterated
- Q8 量化权重约 281G
- Q2 量化权重约 92G

测试硬件：

- GPU: RTX 4070 Laptop 8G
- 内存: 32G DDR5，实际占用约 8G
- 硬盘: Lexar SSD ARES 4TB PCIe 4.0，标称读速 7000M；早期推理读速约 400M，优化后约 1.2GB/s，峰值约 1.5GB/s
- CPU: Intel i9-13900HS

实测结果：

- 模型加载约 1 分钟
- Q2: 约 0.12 token/s（早期未优化参考结果）
- Q8 早期基线: 约 0.06 token/s（未优化）
- Q8 当前优化后: 长输出后段约 0.35 token/s，约 6 倍速度提升

这已经不只是“跑通流程”了。当前版本针对 Windows + NVMe + Disk MoE 做了一轮实打实的优化：并发读、offset ReadFile、热权重缓存、线程参数和本地 OpenAI API 长流式路径都已经打通。Q8 从最早约 0.06 token/s 提升到约 0.35 token/s，更适合一晚上慢慢跑出一篇长文章。优化后推理读盘已经能到约 1.2GB/s，峰值约 1.5GB/s；CPU 占用也下降了，不再必须长期顶到 100%。

另外，当前版本重点修复了 DeepSeek V4 在 128 KV/past cache 边界附近会导致服务端断流的问题。现在长生成不再被这个 bug 提前截断，可以继续压到 KV cache 的真实极限，适合长篇小说这类长输出任务，跑到你存储和 KV cache 能撑住的极限。

<p>
  <img src="效果/Q2.png" alt="Q2 推理效果" width="48%">
  <img src="效果/Q2-硬盘.png" alt="Q2 硬盘占用" width="48%">
</p>

<p>
  <img src="效果/Q8.png" alt="Q8 推理效果" width="48%">
  <img src="效果/Q8-硬盘.png" alt="Q8 硬盘占用" width="48%">
</p>

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

## DeepSeek V4 Pro Q4_K_M 适配入口

新模型目录：

```text
F:\model\DeepSeek-V4-Pro-GGUF-Q4_K_M
```

这个目录是 21 分片 GGUF。当前运行层会把目录自动解析为首分片：

```text
deepseek-ai-DeepSeek-V4-Pro-Q4_K_M-00001-of-00021.gguf
```

推荐使用已经验证能生成 token 的激进硬盘模式。它只启动 server，不自动加载模型：

```powershell
.\run_deepseek_v4_pro_aggressive_disk.bat
```

检查路径和分片状态：

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:9000/health -Method Get
```

确认 `model_files.missing_files` 为空后，再手动加载：

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

这次取舍是：主计算设备放到 CUDA，MoE 权重放硬盘，给硬盘 MoE 4GB 缓存，并把 GPU 显存比例提高到 0.85。这样不是“最省资源”，而是优先保证能完成加载、prefill 和生成 token。实际运行时显存会接近占满，内存也会被明显吃满，但系统仍能保持可操作，并且最终生成了 `能跑`。

之前试过更保守的极限硬盘模式，把主设备压到 CPU、硬盘缓存压得很小，内存占用确实低，但加载长时间卡在 0%，实际不可用。也试过更激进的接近全量占用配置，例如更高 GPU 比例、更大 KV cache、更大硬盘缓存和更多线程，加载能推进，但系统会进入严重换页，PowerShell 和状态查询都会长时间阻塞。现在这组参数是中间点：允许吃资源，但不把机器压到完全失控。

注意：这个 Pro Q4_K_M 权重约 900GB。虽然 MoE 权重走硬盘，但 fastllm 仍然需要把密集权重、元数据、KV cache、运行时状态和部分缓存放进 RAM/VRAM，所以它不是纯粹“模型多大都只吃硬盘”的 out-of-core 模式。当前结论是：在这台机器上可以跑起来并生成 token，但速度非常慢，适合作为验证和长时间后台任务，不适合作为高交互聊天入口。进程退出时可能看到 CUDA runtime unloading 相关的释放显存报错；这发生在生成完成后的清理阶段，不影响本次 `能跑` 输出结论。

如果只是做路径和分片状态诊断，也可以使用较保守的 server 入口：

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
