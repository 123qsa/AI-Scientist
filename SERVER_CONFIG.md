# 云端服务器配置

## SSH 连接信息

```bash
ssh -i ~/Desktop/服务器公私钥/id_ed25526574_qq_com -p 22 hanjiajun@166.111.86.21
```

## 服务器用途

用于运行 AI-Scientist 实验（需要 GPU/CUDA 支持）

## 快速连接脚本

创建 `connect_server.sh`:

```bash
#!/bin/bash
ssh -i ~/Desktop/服务器公私钥/id_ed25526574_qq_com -p 22 hanjiajun@166.111.86.21
```

## 文件传输

上传文件到服务器:
```bash
scp -i ~/Desktop/服务器公私钥/id_ed25526574_qq_com -P 22 \
    local_file.txt hanjiajun@166.111.86.21:/remote/path/
```

下载文件:
```bash
scp -i ~/Desktop/服务器公私钥/id_ed25526574_qq_com -P 22 \
    hanjiajun@166.111.86.21:/remote/path/file.txt ./
```

## 服务器部署步骤

1. 连接服务器
2. 克隆项目: `git clone https://github.com/123qsa/AI-Scientist.git`
3. 创建 conda 环境: `conda create -n ai_scientist python=3.11`
4. 安装依赖: `pip install -r requirements.txt`
5. 准备 baseline: `cd templates/nanoGPT_lite && python experiment.py --out_dir run_0`
6. 运行实验: `python launch_scientist.py --model kimi-k2.5 --experiment nanoGPT_lite`

## 注意事项

- 服务器需要 NVIDIA GPU + CUDA 环境
- 需要配置 API Keys（Kimi OAuth 或 API Key）
