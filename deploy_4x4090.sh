#!/bin/bash
# 4x RTX 4090 服务器部署脚本

set -e

echo "=========================================="
echo "  CVPR-Auto 4x RTX 4090 部署脚本"
echo "=========================================="

# 检查GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi 未找到，请安装NVIDIA驱动"
    exit 1
fi

echo ""
echo "📊 检测到GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1)
echo ""
echo "📊 GPU数量: $GPU_COUNT"

if [ "$GPU_COUNT" -lt 4 ]; then
    echo "⚠️ 警告: 检测到 $GPU_COUNT 张GPU，预期 4 张"
fi

# 检查CUDA
echo ""
echo "🔍 检查CUDA..."
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA未安装"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo "✓ CUDA版本: $CUDA_VERSION"

# 创建虚拟环境
echo ""
echo "📦 创建虚拟环境..."
python3 -m venv venv_4090
source venv_4090/bin/activate

# 升级pip
pip install --upgrade pip setuptools wheel

# 安装PyTorch (CUDA 12.1，4090推荐)
echo ""
echo "📦 安装PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# 验证PyTorch GPU
echo ""
echo "🔍 验证PyTorch GPU..."
python3 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'CUDA版本: {torch.version.cuda}')
print(f'GPU数量: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# 安装 Detectron2 (编译安装，针对4090优化)
echo ""
echo "📦 安装Detectron2..."
# 设置环境变量优化编译
export TORCH_CUDA_ARCH_LIST="8.9"  # 4090的架构
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 安装MMSegmentation
echo ""
echo "📦 安装MMSegmentation..."
pip install -U openmim
mim install mmengine mmcv
pip install mmsegmentation

# 安装CVPR-Auto依赖
echo ""
echo "📦 安装CVPR-Auto..."
pip install -r requirements.txt
pip install gradio optuna tensorboardX

# 安装性能监控工具
echo ""
echo "📦 安装监控工具..."
pip install gpustat py3nvml

# 创建数据目录
echo ""
echo "📁 创建数据目录..."
mkdir -p /data/imagenet
mkdir -p /data/coco
mkdir -p /data/ade20k
mkdir -p /checkpoints
mkdir -p /results

# 创建启动脚本
echo ""
echo "📝 创建启动脚本..."
cat > /start_training.sh << 'EOF'
#!/bin/bash
# 启动分布式训练

cd /workspace/AI-Scientist
source venv_4090/bin/activate

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 如果不使用InfiniBand

# 运行训练
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_addr=localhost \
    --master_port=29500 \
    cvpr_auto/main.py \
    --dataset imagenet \
    --distributed \
    --config configs/server_4090_4x.toml
EOF

chmod +x /start_training.sh

# 创建监控脚本
cat > /monitor.sh << 'EOF'
#!/bin/bash
# GPU监控脚本
while true; do
    clear
    echo "=== GPU Status ==="
    gpustat
    echo ""
    echo "=== Memory ==="
    free -h
    sleep 5
done
EOF

chmod +x /monitor.sh

echo ""
echo "=========================================="
echo "  ✅ 部署完成!"
echo "=========================================="
echo ""
echo "快速开始:"
echo "  1. 激活环境: source venv_4090/bin/activate"
echo "  2. 监控GPU: ./monitor.sh"
echo "  3. 开始训练: ./start_training.sh"
echo "  4. 或运行: python -m cvpr_auto.main --dataset imagenet --parallel 4"
echo ""
echo "数据路径:"
echo "  ImageNet: /data/imagenet"
echo "  COCO: /data/coco"
echo "  ADE20K: /data/ade20k"
echo ""
echo "检查点保存: /checkpoints"
echo "结果输出: /results"
