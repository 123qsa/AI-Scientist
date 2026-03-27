# AI-Scientist 云端版使用指南

## 架构

```
┌─────────────────┐      SSH       ┌─────────────────────────────┐
│   本地控制台     │  ═══════════►  │      云端服务器              │
│  (Web UI/CLI)   │                │  (166.111.86.21, GPU/CUDA)  │
└─────────────────┘                └─────────────────────────────┘
        │                                    │
        │ 显示输出/同步结果                    │ 运行实验
        ▼                                    ▼
   results/ 目录                        完整环境
```

## 快速开始

### 1. 启动 Web UI

```bash
# 安装依赖
pip install gradio

# 启动界面
python web_ui.py
```

访问: http://localhost:7860

### 2. 命令行运行

```bash
# 远程执行（默认）
python launch_scientist_remote.py --experiment nanoGPT_lite --num-ideas 2

# 本地调试
LOCAL_MODE=1 python launch_scientist_remote.py
```

### 3. 一键连接服务器

```bash
./connect_server.sh
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `web_ui.py` | Web 界面（云端执行） |
| `launch_scientist_remote.py` | 命令行入口（云端执行） |
| `remote_runner.py` | 远程执行核心模块 |
| `launch_scientist.py` | 原版（本地执行） |
| `SERVER_CONFIG.md` | 服务器配置详情 |
| `connect_server.sh` | SSH 连接脚本 |

## 服务器要求

- **GPU**: NVIDIA GPU + CUDA
- **网络**: 校园网/VPN
- **存储**: 实验结果保存在服务器，自动同步到本地

## 环境变量

```bash
# 强制本地运行（调试）
export LOCAL_MODE=1

# OpenAlex 邮箱（用于文献检索）
export OPENALEX_MAIL_ADDRESS="your@email.com"
```

## 常见问题

**Q: 无法连接服务器？**
A: 检查 VPN 和 SSH 密钥配置

**Q: 实验结果在哪里？**
A: 服务器运行后自动同步到本地 `results/` 目录

**Q: 如何查看服务器日志？**
A: SSH 登录后查看 `~/AI-Scientist/results/`
