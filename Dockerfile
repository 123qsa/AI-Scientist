# CVPR-Auto Docker 镜像
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    vim \
    ssh \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /workspace

# 复制项目文件
COPY . /workspace/

# 安装 Python 依赖
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install -r requirements.txt && \
    pip3 install gradio optuna

# 安装 Detectron2
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'

# 创建输出目录
RUN mkdir -p /workspace/cvpr_outputs

# 暴露端口（用于 Web UI）
EXPOSE 7860

# 默认命令
CMD ["python3", "-m", "cvpr_auto.main", "--help"]
