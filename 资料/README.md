# 当前项目整理说明

这个目录放的是资料、参考代码、旧脚本和调试日志；根目录只保留真正跑模型或重编译会用到的东西。

## 根目录里真正有用的东西

- `fastllm-master/`：当前实际使用的框架源码和已编译产物。DeepSeek V4 GGUF 的适配补丁都在这里。
- `cuda-12.4-toolkit/`：当前运行和重编译用到的 CUDA 12.4 工具链。`build_cuda_fastllm.cmd` 和运行脚本都会用它。
- `run_q2_generate_chat.py`：单次命令行测试入口，当前默认跟随 `model_runtime.py` 加载 Q8 GGUF。
- `model_runtime.py`：模型加载、卸载和生成的公共运行层。
- `openai_compatible_server.py`：OpenAI 兼容 API 服务，默认 `127.0.0.1:9000`。
- `probe_q2_dsv4.py`：排查 DeepSeek V4 GGUF 加载、结构识别、设备映射时用的辅助脚本。
- `build_cuda_fastllm.cmd`：改完 `fastllm-master` 后重新编译用。

模型文件不在这个项目目录里，当前脚本默认读取：

```powershell
F:\下载\cyberneurova-DeepSeek-V4-Flash-abliterated-Q8_0.gguf
```

## 现在怎么跑

在项目根目录执行：

```powershell
$env:DSV4_THREADS='32'
$env:DSV4_MAX_NEW_TOKENS='64'
python .\run_q2_generate_chat.py "你好，请用中文回答：一加一等于几？"
```

已经验证过 Q2 能正常输出中文。成功记录放在：

- `资料/验证结果/run_q2_final_chinese.log`
- `资料/验证结果/run_q2_final_intro.log`

## OpenAI 兼容 API

启动服务：

```powershell
python .\openai_compatible_server.py
```

客户端配置：

```text
Base URL: http://127.0.0.1:9000/v1
API Key: local
Model: deepseek-v4-flash-q8
```

接口：

- `GET /health`：查看模型是否已加载、线程数、KV cache 等状态。
- `GET /v1/models`：返回本地模型列表。
- `POST /v1/chat/completions`：OpenAI 风格聊天接口，支持 `stream: true` 和 `stream: false`。
- `POST /admin/load`：手动加载模型。
- `POST /admin/unload`：手动卸载模型。

## 关键注意点

- 不要在当前脚本里加回 `model.set_moe_experts(1)`。这个会把 DeepSeek V4 的 MoE 路由搞坏，之前会导致输出乱码或不正常。
- 当前默认模型已切换为 Q8_0 GGUF。之前确认跑通的是 Q2_K；Q8 需要重新启动 server 后实际验证。
- `cuda-12.4-extract/` 只是 CUDA 解包过程留下的构建材料，不参与日常运行，已经放到 `资料/构建材料/`。

## 资料目录里分别是什么

- `教程与上下文/`：最开始看的部署教程文本。
- `Codex会话导出/`：之前要求导出的 Codex 上下文记录。
- `参考源码/ds4/`：antirez 的 ds4 参考实现。它推荐 Apple/Metal 路线，但这里只是参考，不参与当前 fastllm 运行。
- `参考源码/fastllm-upstream/`：上游 fastllm 参考仓库，用来对比改动，不参与当前运行。
- `旧脚本/`：调试过程中写过的旧入口，正常跑模型不用它们。
- `调试日志/build/`：多次编译日志。
- `调试日志/probe/`：结构识别、加载排查日志。
- `调试日志/run/`：各种旧运行尝试日志。
- `验证结果/`：最终确认能正常输出的日志。

## 当前适配大概做了什么

- 补了 DeepSeek V4 / `deepseek4` / `deepseek_v4` 的 GGUF 架构和权重名映射。
- 补了 DeepSeek V4 GGUF 特殊 token 读取。
- 调整了磁盘 MoE 权重加载，让大部分 MoE 权重能走硬盘映射，减少内存压力。
- 保留 shared experts 在内存/正常设备上，避免错误地把共享专家当普通磁盘 MoE 权重处理。
- 修了几处 Q2/Q8 GGUF 路径下 DeepSeek V4 推理会出异常结果的问题。
