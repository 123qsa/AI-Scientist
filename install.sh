#!/bin/bash
# CVPR-Auto 安装脚本

set -e

echo "=========================================="
echo "  CVPR-Auto Installation Script"
echo "=========================================="

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✓ Python version: $PYTHON_VERSION"

# 创建虚拟环境
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 升级 pip
pip install --upgrade pip

# 安装基础依赖
echo ""
echo "📦 Installing base dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 CV 库
echo ""
echo "📦 Installing computer vision libraries..."
pip install opencv-python pillow matplotlib seaborn scikit-learn scikit-image

# 安装 Detectron2 (检测)
echo ""
echo "📦 Installing Detectron2..."
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 安装 MMSegmentation (分割)
echo ""
echo "📦 Installing MMSegmentation..."
pip install openmim
mim install mmengine mmcv
pip install mmsegmentation

# 安装项目依赖
echo ""
echo "📦 Installing project dependencies..."
pip install -r requirements.txt

# 安装 CVPR-Auto 依赖
echo ""
echo "📦 Installing CVPR-Auto dependencies..."
pip install gradio optuna anthropic openai

# 设置环境变量
echo ""
echo "⚙️  Setting up environment..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
# LLM Configuration
LLM_PROVIDER=kimi
ANTHROPIC_API_KEY=
OPENAI_API_KEY=

# Server Configuration
SERVER_IP=166.111.86.21
SERVER_USER=hanjiajun

# OpenAlex
OPENALEX_MAIL_ADDRESS=
EOF
    echo "✓ Created .env file. Please edit it with your API keys."
fi

# 创建必要的目录
echo ""
echo "📁 Creating directories..."
mkdir -p cvpr_outputs
mkdir -p data/imagenet
mkdir -p data/coco
mkdir -p data/ade20k

echo ""
echo "=========================================="
echo "  ✅ Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys"
echo "  2. Run: source venv/bin/activate"
echo "  3. Run: ./run_cvpr_auto.sh cifar10"
echo ""
