# CVPR-Auto: 全自动 CVPR 级别论文生成系统

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         CVPR-Auto System                         │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: Idea Generation & Novelty Verification                │
│    ├── Track latest CVPR/ICCV/ECCV/NeurIPS papers               │
│    ├── Generate novel ideas with theoretical backing            │
│    └── Automatic novelty verification (citation network)        │
│                                                                  │
│  Phase 2: Large-Scale Experimentation                           │
│    ├── ImageNet-1K / COCO / ADE20K support                      │
│    ├── Distributed multi-GPU training                           │
│    ├── Automatic hyperparameter search (Optuna)                 │
│    └── Comprehensive ablation studies                           │
│                                                                  │
│  Phase 3: Professional Paper Composition                        │
│    ├── CVPR official LaTeX template                             │
│    ├── Professional figure generation (matplotlib/PGF)          │
│    ├── Structured writing (per-section optimization)            │
│    └── Supplementary material generation                        │
│                                                                  │
│  Phase 4: Self-Review & Iterative Improvement                   │
│    ├── Simulate reviewer perspective                            │
│    ├── Identify weak points (experiments/writing/novelty)       │
│    └── Auto-improvement (3-5 iterations)                        │
│                                                                  │
│  Phase 5: Quality Assessment                                    │
│    ├── Multi-dimensional scoring                                │
│    ├── Compare with CVPR historical papers                      │
│    └── Quality gate (threshold-based output)                    │
└─────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 一键运行（推荐）

```bash
# 快速测试模式 (CIFAR-10, ~2小时)
./run_cvpr_auto.sh cifar10

# 完整模式 (ImageNet, ~3-7天)
./run_cvpr_auto.sh imagenet
```

### 手动运行

```bash
# 使用 Python 模块
python -m cvpr_auto.main \
    --dataset imagenet \
    --task classification \
    --num-ideas 10 \
    --max-revision-rounds 5

# 快速测试
python -m cvpr_auto.main --quick-test
```

## 项目结构

```
cvpr_auto/
├── __init__.py           # 包初始化
├── config.py             # 全局配置（数据集、阈值、服务器）
├── main.py               # 主入口
├── remote_runner.py      # 云端服务器执行
├── experiment_base.py    # CVPR 实验基类
├── hyperparam_search.py  # 超参搜索 & 消融实验
├── paper_composer.py     # LaTeX 论文生成 & 图表
└── self_review.py        # 自评审 & 质量关卡

templates/cvpr/           # CVPR 专用模板（待创建）
├── classification/
├── detection/
└── segmentation/
```

## 配置说明

编辑 `cvpr_auto/config.py`:

```python
# 服务器配置
SERVER_IP = "166.111.86.21"
SERVER_USER = "hanjiajun"
SSH_KEY = "~/Desktop/服务器公私钥/id_ed25526574_qq_com"

# 数据集路径（服务器端）
DATASETS = {
    "imagenet": {"path": "/data/imagenet", "num_classes": 1000},
    "coco": {"path": "/data/coco", "tasks": ["detection", "instance_seg"]},
    "ade20k": {"path": "/data/ade20k"},
}

# CVPR 质量阈值
QUALITY_THRESHOLDS = {
    "novelty_score": 7.5,      # 创新性
    "experiment_rigor": 8.0,    # 实验严谨性
    "writing_quality": 7.5,     # 写作质量
    "min_improvement": 1.0,     # 相比 SOTA 提升
    "min_ablations": 5,         # 消融实验数
    "min_datasets": 2,          # 数据集数
}
```

## 输出文件

```
cvpr_outputs/
├── paper.pdf                    # 主论文
├── paper.tex                    # LaTeX 源文件
├── supplementary.pdf            # 补充材料
├── figures/
│   ├── fig1_training_curves.pdf
│   ├── fig2_sota_comparison.pdf
│   ├── fig3_ablation.pdf
│   └── fig4_hyperparam.pdf
├── tables/
│   └── ablation_table.tex
├── code/                        # 可复现代码
├── review_report.json           # 自评审报告
└── experiment_logs/             # 实验日志
```

## 关键特性

| 特性 | 说明 |
|------|------|
| **大规模支持** | ImageNet-1K, COCO, ADE20K |
| **分布式训练** | 多 GPU DataParallel |
| **自动搜索** | Optuna 超参优化 |
| **完整消融** | 自动生成消融实验配置 |
| **专业图表** | CVPR 格式高质量图表 |
| **自评审** | 模拟审稿人多轮改进 |
| **质量关卡** | 达到阈值才输出论文 |

## 工作流程

```
Day 1-2:  想法生成 & 验证（AI自动）
Day 3-5:  大规模实验（云端运行）
Day 6:    论文撰写（AI生成 + 图表）
Day 7:    自评审 & 改进迭代
Day 8+:   人工审核 & 最终润色
```

## 人工审核清单

AI 生成后必须人工检查：

- [ ] 实验正确性（代码 review）
- [ ] 对比公平性（baseline 是否复现正确）
- [ ] 创新性描述（是否准确）
- [ ] 相关工作完整性
- [ ] 图表质量（分辨率、字体）
- [ ] 数学公式正确性
- [ ] 引用准确性

## 注意事项

1. **资源需求**
   - ImageNet 实验需要 4x V100 或更高
   - 预计耗时 3-7 天
   - 成本约 $500-1000

2. **成功率**
   - 直接生成投 CVPR：~5%
   - AI + 深度人工改进：~30-50%

3. **伦理考虑**
   - 需声明 AI 辅助生成
   - 作者需对内容负责
   - 建议先投 ArXiv 获取反馈

## 后续改进方向

- [ ] 集成 detectron2 / mmdetection
- [ ] 支持更多任务（3D vision, video）
- [ ] 自动 code review
- [ ] 与 OpenReview 集成
- [ ] 多智能体协作系统
