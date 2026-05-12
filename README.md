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
- 硬盘: Lexar SSD ARES 4TB PCIe 4.0，标称读速 7000M，实测推理读速约 400M
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
cd C:\Users\user\output\deepseek-v4部署
```

启动 OpenAI 兼容 server：

```powershell
python .\openai_compatible_server.py
```

这条命令会自动做两件事：

1. 加载模型：

```text
F:\下载\cyberneurova-DeepSeek-V4-Flash-abliterated-Q8_0.gguf
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
