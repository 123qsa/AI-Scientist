# CVPR-Auto 迭代改进机制详解

## 概述

CVPR-Auto 的核心优势在于其**完整的迭代改进闭环**：

```
生成 → 实验 → 评审 → 改进 → 验证 → (循环直到达标)
```

## 迭代流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     Iteration Controller                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Round 1: Initial Generation                                     │
│    ├─ 生成初始想法                                                │
│    ├─ 运行完整实验 (数据集+超参搜索+消融)                          │
│    ├─ 生成论文初稿                                                │
│    └─ 自评审 → Score: 6.5/10                                     │
│         ❌ 低于阈值 7.5, 继续迭代                                  │
│                                                                  │
│  Round 2: Experiment Improvement                                 │
│    ├─ 分析弱点: 消融实验不足 (2/5), 缺少复杂度分析                 │
│    ├─ 生成改进方案:                                               │
│    │   • 添加 3 个消融实验                                        │
│    │   • 添加 FLOPs/Params 分析                                   │
│    │   • 添加可视化结果                                           │
│    ├─ 执行改进 → 运行新实验                                       │
│    ├─ 更新论文内容                                                │
│    └─ 自评审 → Score: 7.8/10                                     │
│         ✅ 达到阈值, 但继续迭代寻求更好                            │
│                                                                  │
│  Round 3: Writing Improvement                                    │
│    ├─ 分析弱点: Introduction 太短, Method 不够详细                │
│    ├─ 生成改进方案:                                               │
│    │   • 扩展 Introduction (+200 words)                           │
│    │   • 添加 explicit contributions list                         │
│    │   • 添加 complexity analysis section                         │
│    ├─ LLM 改进写作                                                │
│    └─ 自评审 → Score: 8.2/10                                     │
│         ✅ 质量达标, 停止迭代                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. SelfReviewer（自评审器）

**功能**：模拟审稿人视角，多维度评估论文

**评估维度**：
- `novelty`: 创新性 (理论贡献、方法新颖度)
- `experiment_rigor`: 实验严谨性 (数据集数量、消融实验、超参搜索)
- `writing_quality`: 写作质量 (清晰度、逻辑性、完整性)
- `significance`: 重要性 (相比 SOTA 的提升、影响力)
- `overall`: 综合评分

**示例输出**：
```json
{
  "scores": {
    "novelty": 7.5,
    "experiment_rigor": 6.0,
    "writing_quality": 7.0,
    "significance": 7.5,
    "overall": 7.0
  },
  "weaknesses": [
    "Ablation study insufficient (2/5 required)",
    "Missing complexity analysis (FLOPs/Params)",
    "Introduction too short (< 400 words)"
  ],
  "suggestions": [
    "Add ablations on attention mechanism and normalization",
    "Include computational complexity comparison table",
    "Expand introduction with more motivation"
  ]
}
```

### 2. ImprovementGenerator（改进生成器）

**功能**：根据评审结果生成具体改进方案

**改进类型**：

#### 实验改进 (Experiment Improvements)

| 类型 | 触发条件 | 改进行动 |
|------|---------|---------|
| `add_experiment/ablation` | 消融实验 < 5 个 | 设计并运行新的消融实验 |
| `add_experiment/dataset` | 数据集 < 2 个 | 在额外数据集上验证 |
| `add_experiment/hyperparam` | 无超参搜索 | 运行 Optuna 搜索 |
| `add_experiment/visualization` | 缺少可视化 | 生成 attention 图、特征可视化 |
| `add_experiment/robustness` | 无鲁棒性测试 | 添加 corruption/noise 测试 |
| `add_analysis/complexity` | 无复杂度分析 | 计算 FLOPs/Params |

#### 写作改进 (Paper Improvements)

| 类型 | 触发条件 | 改进行动 |
|------|---------|---------|
| `modify_writing/clarity` | 写作分数 < 7 | 改进句子结构、逻辑 flow |
| `expand_section/introduction` | Introduction < 400 词 | 添加背景、motivation |
| `add_content/contributions` | 无 explicit contributions | 添加 bullet point 贡献列表 |
| `expand_section/method` | Method < 600 词 | 添加实现细节、伪代码 |
| `add_content/complexity_analysis` | 无复杂度分析 | 添加 FLOPs/Params section |
| `add_content/limitations` | 无 limitations | 添加局限性讨论 |

### 3. IterationController（迭代控制器）

**功能**：协调整个迭代流程，管理状态

**关键方法**：

```python
# 运行完整迭代循环
final_paper, final_experiments, success = controller.run_iteration_loop(
    initial_idea=idea,
    initial_code=code,
    output_dir=output_dir
)

# 获取改进报告
report = controller.get_improvement_report()
```

**停止条件**（满足任一）：
1. ✅ 综合评分 ≥ 阈值 (默认 7.5)
2. ⚠️ 达到最大迭代轮数 (默认 5 轮)
3. ⚠️ 连续两轮无改进

## 状态追踪

每次迭代记录 `IterationState`：

```python
@dataclass
class IterationState:
    round_num: int           # 第几轮
    timestamp: str          # 时间戳
    review_scores: Dict     # 评审分数
    weaknesses: List[str]   # 本轮弱点
    improvements_made: List[str]  # 执行的改进
    experiment_delta: Dict  # 实验指标变化
    paper_delta: Dict       # 论文变化
```

**输出文件**：`iteration_history.json`

```json
[
  {
    "round_num": 1,
    "timestamp": "2024-03-27T10:00:00",
    "review_scores": {
      "novelty": 7.0,
      "experiment_rigor": 5.5,
      "writing_quality": 6.5,
      "overall": 6.3
    },
    "weaknesses": ["Ablation insufficient", "Missing complexity analysis"],
    "improvements_made": [],
    "experiment_delta": {},
    "paper_delta": {}
  },
  {
    "round_num": 2,
    "timestamp": "2024-03-27T14:00:00",
    "review_scores": {
      "novelty": 7.5,
      "experiment_rigor": 8.0,
      "writing_quality": 7.0,
      "overall": 7.6
    },
    "weaknesses": ["Introduction needs expansion"],
    "improvements_made": [
      "Added 3 ablation experiments",
      "Added FLOPs/Params analysis",
      "Added attention visualizations"
    ],
    "experiment_delta": {
      "experiment_rigor_delta": 2.5
    },
    "paper_delta": {
      "writing_quality_delta": 0.5
    }
  }
]
```

## 质量阈值配置

编辑 `cvpr_auto/config.py`：

```python
QUALITY_THRESHOLDS = {
    # 各项最低分数 (1-10)
    "novelty_score": 7.5,
    "experiment_rigor": 8.0,
    "writing_quality": 7.5,
    "significance": 7.0,

    # 实验要求
    "min_improvement": 1.0,     # 相比 SOTA 最低提升 (%)
    "min_ablations": 5,          # 最少消融实验数
    "min_datasets": 2,           # 最少数据集数
    "min_baselines": 5,          # 最少对比 baseline 数
}

MAX_REVISION_ROUNDS = 5  # 最大迭代轮数
```

## 使用示例

### 快速测试（2轮迭代）

```bash
python -m cvpr_auto.main --quick-test
```

**预期输出**：
```
==================================================
🚀 CVPR-Auto: Fully Automated CVPR Paper Generation
   With Complete Iterative Improvement
==================================================

⚡ Quick Test Mode Enabled
📁 Output Directory: ./cvpr_outputs/cifar10_20240327_120000

☁️  Checking cloud server connection...
✅ Connected to 166.111.86.21

📋 Configuration:
   Dataset: cifar10
   Task: classification
   Max Iterations: 2
   Quality Threshold: 7.5/10

==================================================
🎯 Phase 1: Idea Generation & Novelty Verification
--------------------------------------------------
Generating 2 candidate ideas...
  Idea 1: Novel Architecture for classification on cifar10
  Idea 2: Improved Attention Mechanism for classification on cifar10

✅ Selected Idea: Novel Architecture for classification on cifar10

==================================================
📋 Iteration Round 1/2
==================================================

🔬 Running Experiments...
Epoch 10: Train Acc=82.34%, Val Acc=78.56%

📝 Generating Paper...
  Generating figures...
  Generating LaTeX...

🔍 Self-reviewing...

📊 Review Scores:
  ✅ novelty: 7.5/10 (threshold: 7.5)
  ⚠️ experiment_rigor: 6.0/10 (threshold: 8.0)
  ✅ writing_quality: 7.0/10 (threshold: 7.5)
  ⚠️ overall: 6.8/10 (threshold: 7.5)

🔧 Generating improvements...

📋 Planned Improvements (3 items):
  1. [HIGH] Add 3 more ablation studies
  2. [HIGH] Add computational complexity analysis
  3. [MEDIUM] Add qualitative visualizations

💾 Checkpoint saved to ./cvpr_outputs/.../checkpoints/round_1.json

==================================================
📋 Iteration Round 2/2
==================================================

🔬 Running additional experiments...
  Applying: Add 3 more ablation studies
  Running: w/o attention, w/o normalization, w/o augmentation

📝 Updating paper...
  Applying: Add computational complexity analysis

🔍 Self-reviewing...

📊 Review Scores:
  ✅ novelty: 7.5/10 (threshold: 7.5)
  ✅ experiment_rigor: 8.5/10 (threshold: 8.0)
  ✅ writing_quality: 7.5/10 (threshold: 7.5)
  ✅ overall: 7.9/10 (threshold: 7.5)

✅ Quality gate passed! Final score: 7.9

💾 Iteration history saved to ./cvpr_outputs/.../iteration_history.json

==================================================
📊 ITERATION IMPROVEMENT REPORT
==================================================

Total Rounds: 2

Score Progression:
  novelty              : 7.5 → 7.5 (→0.0)
  experiment_rigor     : 6.0 → 8.5 (↑2.5)
  writing_quality      : 7.0 → 7.5 (↑0.5)
  significance         : 7.0 → 7.5 (↑0.5)
  overall              : 6.8 → 7.9 (↑1.1)

Improvements Made:
  Round 1:
  Round 2:
    - Add 3 more ablation studies
    - Add computational complexity analysis
    - Add qualitative visualizations

==================================================
🎉 SUCCESS: Quality threshold met!
==================================================

📂 Output Files:
  ./cvpr_outputs/cifar10_20240327_120000/paper.tex
  ./cvpr_outputs/cifar10_20240327_120000/figures/
  ./cvpr_outputs/cifar10_20240327_120000/iteration_history.json
  ./cvpr_outputs/cifar10_20240327_120000/checkpoints/
```

### 完整运行（5轮迭代）

```bash
python -m cvpr_auto.main \
    --dataset imagenet \
    --num-ideas 10 \
    --max-revision-rounds 5 \
    --min-quality-score 8.0
```

## 高级用法

### 从检查点继续

```bash
python -m cvpr_auto.main \
    --continue-from ./cvpr_outputs/imagenet_xxx/checkpoints/round_3.json
```

### 强制完整迭代

即使提前达标，也跑完所有轮数：

```bash
python -m cvpr_auto.main --force-full-iterations
```

### 自定义质量阈值

```bash
python -m cvpr_auto.main \
    --min-quality-score 8.5 \
    --max-revision-rounds 10
```

## 调试与监控

### 查看迭代历史

```bash
cat cvpr_outputs/xxx/iteration_history.json | jq '.[] | {round: .round_num, score: .review_scores.overall}'
```

### 对比不同轮次

```bash
# Round 1 vs Round 3 的实验指标
diff <(jq '.[0].review_scores' iteration_history.json) \
     <(jq '.[2].review_scores' iteration_history.json)
```

### 检查改进效果

```python
import json

with open('iteration_history.json') as f:
    history = json.load(f)

for state in history:
    print(f"Round {state['round_num']}: {state['improvements_made']}")
```

## 最佳实践

1. **先跑 Quick Test**: 验证流程再跑完整实验
2. **检查每轮改进**: 确认改进是有效的
3. **调整阈值**: 根据领域调整质量要求
4. **保存检查点**: 可以从任意轮次继续
5. **人工最终审核**: 即使达标，仍需人工 review

## 与其他组件的关系

```
IterationController (主控)
    ├─ SelfReviewer (评审)
    ├─ ImprovementGenerator (生成改进方案)
    ├─ Experiment Runner (执行实验改进)
    │   └─ HyperParamSearcher (超参搜索)
    │   └─ AutoAblation (消融实验)
    ├─ Paper Composer (执行写作改进)
    │   └─ CVPRPaperComposer (论文生成)
    └─ QualityGate (质量把关)
```
