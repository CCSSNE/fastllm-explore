# fastllm Explore

用硬盘来推理 DeepSeek V4。

本仓库提供一个本地 OpenAI 兼容 server，用于加载 DeepSeek V4 Flash GGUF 模型并暴露 `/v1` API。

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
