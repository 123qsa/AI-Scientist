#!/bin/bash
# CVPR-Auto 一键运行脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "=================================================="
echo "  CVPR-Auto: Fully Automated Paper Generation"
echo "=================================================="
echo -e "${NC}"

# 检查参数
MODE=${1:-"cifar10"}  # 默认使用 CIFAR-10 快速测试

if [ "$MODE" == "imagenet" ]; then
    echo -e "${YELLOW}⚠️  Warning: ImageNet experiments require significant GPU resources${NC}"
    echo "Estimated time: 3-7 days on 4x V100 GPUs"
    echo "Estimated cost: ~$500-1000"
    echo ""
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    DATASET="imagenet"
    NUM_IDEAS=10
elif [ "$MODE" == "quick" ] || [ "$MODE" == "cifar10" ]; then
    echo -e "${GREEN}⚡ Quick test mode (CIFAR-10)${NC}"
    DATASET="cifar10"
    NUM_IDEAS=3
else
    echo "Usage: $0 [cifar10|imagenet]"
    echo ""
    echo "Options:"
    echo "  cifar10   - Quick test with CIFAR-10 (~2 hours)"
    echo "  imagenet  - Full ImageNet experiments (~3-7 days)"
    exit 1
fi

# 检查环境
echo -e "${BLUE}Checking environment...${NC}"

if [ ! -f "cvpr_auto/main.py" ]; then
    echo -e "${RED}❌ Error: cvpr_auto/main.py not found${NC}"
    echo "Please run from project root directory"
    exit 1
fi

# 检查 SSH 密钥
if [ ! -f "$HOME/Desktop/服务器公私钥/id_ed25526574_qq_com" ]; then
    echo -e "${RED}❌ Error: SSH key not found${NC}"
    exit 1
fi

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Error: Python 3 not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Environment check passed${NC}"

# 运行 CVPR-Auto
echo ""
echo -e "${BLUE}Starting CVPR-Auto...${NC}"
echo ""

python3 -m cvpr_auto.main \
    --dataset $DATASET \
    --num-ideas $NUM_IDEAS \
    --output-dir ./cvpr_outputs/${DATASET}_$(date +%Y%m%d_%H%M%S)

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=================================================="
    echo "  ✅ CVPR Paper Generation Complete!"
    echo "==================================================${NC}"
    echo ""
    echo "Output location: ./cvpr_outputs/"
    echo ""
    echo "Next steps:"
    echo "  1. Review the generated paper.pdf"
    echo "  2. Check supplementary.pdf"
    echo "  3. Verify code reproducibility"
    echo "  4. Consider professional proofreading"
    echo ""
else
    echo ""
    echo -e "${RED}❌ CVPR-Auto failed. Check logs for details.${NC}"
    exit 1
fi
