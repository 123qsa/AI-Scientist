# CVPR-Auto 快速上手指南

## 5 分钟快速开始

### 1. 安装

```bash
# 克隆仓库
git clone https://github.com/123qsa/AI-Scientist.git
cd AI-Scientist

# 运行安装脚本
./install.sh

# 激活环境
source venv/bin/activate
```

### 2. 配置 API Keys

编辑 `.env` 文件：

```bash
# LLM 配置（选择其一）
LLM_PROVIDER=kimi                    # 或 anthropic/openai
ANTHROPIC_API_KEY=sk-ant-xxxxx       # 如果用 Claude
OPENAI_API_KEY=sk-xxxxx              # 如果用 OpenAI

# 文献检索
OPENALEX_MAIL_ADDRESS=your@email.com
```

### 3. 运行快速测试

```bash
# CIFAR-10 快速测试（约 30 分钟）
./run_cvpr_auto.sh cifar10

# 或使用 Python 直接运行
python -m cvpr_auto.main --quick-test
```

## 完整流程示例

### 生成一篇 CVPR 级别的论文

```bash
# 1. ImageNet 分类任务（3-7 天）
python -m cvpr_auto.main \
    --dataset imagenet \
    --task classification \
    --num-ideas 10 \
    --max-revision-rounds 5

# 2. COCO 检测任务
python -m cvpr_auto.main \
    --dataset coco \
    --task detection \
    --num-ideas 5

# 3. ADE20K 分割任务
python -m cvpr_auto.main \
    --dataset ade20k \
    --task segmentation
```

## 输出结构

```
cvpr_outputs/
├── imagenet_20240327_120000/
│   ├── paper.tex              # LaTeX 源文件
│   ├── paper.pdf              # 编译后的 PDF
│   ├── figures/
│   │   ├── fig1_training.pdf
│   │   ├── fig2_sota.pdf
│   │   └── fig3_ablation.pdf
│   ├── tables/
│   │   └── ablation.tex
│   ├── checkpoints/
│   │   ├── round_1.json       # 迭代检查点
│   │   ├── round_2.json
│   │   └── round_3.json
│   ├── iteration_history.json # 完整的迭代历史
│   └── code/                  # 可复现代码
└── ...
```

## 进阶使用

### 使用 Web UI

```bash
python web_ui.py
```

打开浏览器访问 `http://localhost:7860`

### Docker 部署

```bash
# 构建镜像
docker-compose build

# 运行
docker-compose up -d

# 查看日志
docker-compose logs -f
```

### 远程服务器执行

```bash
# 在云端服务器运行实验
python -m cvpr_auto.remote_runner

# 同步结果
rsync -avz hanjiajun@166.111.86.21:~/AI-Scientist/cvpr_outputs/ ./cvpr_outputs/
```

## 自定义配置

编辑 `cvpr_auto/config.py`：

```python
# 质量阈值
QUALITY_THRESHOLDS = {
    "novelty_score": 8.0,        # 提高创新性要求
    "experiment_rigor": 8.5,     # 提高实验要求
    "writing_quality": 8.0,      # 提高写作要求
    "min_improvement": 1.5,      # 最低提升百分比
}

# 最大迭代轮数
MAX_REVISION_ROUNDS = 7

# 数据集路径
DATASETS = {
    "imagenet": {
        "path": "/your/data/path",
        "num_classes": 1000
    }
}
```

## 迭代过程详解

```
第1轮：初始生成
├── 生成 10 个候选想法
├── 评估新颖性和可行性
├── 选择最佳想法
├── 运行初始实验
├── 生成论文初稿
└── 自评审 → Score: 6.5/10

第2轮：实验改进
├── 添加消融实验
├── 添加复杂度分析
├── 更新论文
└── 自评审 → Score: 7.8/10

第3轮：写作改进
├── 扩展 Introduction
├── 添加贡献列表
├── 改进 Method 描述
└── 自评审 → Score: 8.5/10 ✅

最终结果：满足 CVPR 质量阈值
```

## 常见问题

### Q: 运行时间多长？

| 任务 | 数据集 | 时间 | GPU |
|------|--------|------|-----|
| 快速测试 | CIFAR-10 | 30 分钟 | 1x 3080 |
| 标准实验 | ImageNet | 3-5 天 | 4x V100 |
| 检测实验 | COCO | 5-7 天 | 4x V100 |
| 分割实验 | ADE20K | 3-5 天 | 4x V100 |

### Q: 成本多少？

- **LLM API**: ~$20-50 (Kimi OAuth 免费)
- **云服务器**: ~$500-1000 (4x V100 x 5 天)
- **总成本**: ~$500-1000

### Q: 成功率如何？

- **直接生成**: ~5% (投 Workshop)
- + 人工改进: ~30-50% (投 CVPR)
- 建议先投 ArXiv 获取社区反馈

### Q: 如何保证可复现性？

- 所有代码自动保存到 `code/` 目录
- 完整的实验日志
- 随机种子固定
- Docker 镜像封装环境

## 故障排除

### 问题：LLM 调用失败

```bash
# 检查 Kimi CLI
kimi --version

# 检查 API key
export ANTHROPIC_API_KEY=sk-xxxxx
```

### 问题：CUDA 内存不足

```bash
# 减少 batch size
python -m cvpr_auto.main --batch-size 64

# 使用梯度累积
python -m cvpr_auto.main --grad-accum 4
```

### 问题：迭代不收敛

```bash
# 降低质量阈值
python -m cvpr_auto.main --min-quality-score 7.0

# 增加最大迭代轮数
python -m cvpr_auto.main --max-revision-rounds 10
```

## 最佳实践

1. **先跑快速测试**: 验证配置再跑大实验
2. **监控迭代过程**: 检查每轮改进是否合理
3. **保留检查点**: 可以从任意轮次恢复
4. **人工最终审核**: 即使达标也要人工 review
5. **开源代码**: 增加论文可信度

## 获取更多帮助

- 文档: [CVPR_AUTO_README.md](CVPR_AUTO_README.md)
- 迭代机制: [CVPR_AUTO_ITERATION.md](CVPR_AUTO_ITERATION.md)
- 问题反馈: GitHub Issues

## 示例输出

```
🚀 CVPR-Auto: Fully Automated CVPR Paper Generation
==================================================

📋 Configuration:
   Dataset: imagenet
   Task: classification
   Max Iterations: 5
   Quality Threshold: 7.5/10

🎯 Phase 1: Idea Generation & Novelty Verification
--------------------------------------------------
  Idea 1: architectural_modification + vision_transformers
  Idea 2: loss_function + efficient_network_design
  ...

✓ Generated 10 candidate ideas
✓ Selected: Attention-Augmented EfficientNet

📋 Iteration Round 1/5
==================================================
🔬 Running initial experiments...
📝 Generating paper...
🔍 Self-reviewing...

📊 Review Scores:
  ⚠️ novelty: 7.0/10 (threshold: 7.5)
  ✅ experiment_rigor: 8.0/10 (threshold: 8.0)
  ⚠️ writing_quality: 6.5/10 (threshold: 7.5)

📋 Planned Improvements (5 items):
  1. [HIGH] Add 3 more ablation studies
  2. [HIGH] Expand introduction to >400 words
  ...

==================================================
🎉 SUCCESS: Quality threshold met!
==================================================

📂 Output Files:
  ./cvpr_outputs/imagenet_20240327_120000/paper.tex
  ./cvpr_outputs/imagenet_20240327_120000/paper.pdf
```

---

**Ready to generate your CVPR paper?** Start with `./run_cvpr_auto.sh cifar10`!
