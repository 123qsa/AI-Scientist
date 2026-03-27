# CVPR-Auto 文献驱动的想法生成（增强版）

## 概述

**原版流程的问题**：
```
随机组合改进角度 → 生成想法 → 可能重复已有工作
```

**增强版流程**：
```
抓取最新 100 篇顶会论文 → 分析趋势和空白 → 基于真实 gaps 生成想法
```

## 增强后的完整流程

```
Phase 0: 文献调研（新增 ⭐）
├── 抓取 CVPR/ICCV/ECCV/NeurIPS/ICLR 最新论文
├── LLM 分析每篇论文的问题、方法、局限性
├── 识别研究趋势和空白（Research Gaps）
└── 基于 Gaps 生成具体想法

Phase 1-5: 原有流程（实验/论文/迭代/输出）
```

## 使用方式

### 方式 1: 纯文献驱动（推荐）

```python
from cvpr_auto.paper_tracker import PaperDrivenIdeaPipeline

pipeline = PaperDrivenIdeaPipeline()

# 自动完成文献调研 + 想法生成
gaps, ideas = pipeline.run_full_pipeline(
    venues=['CVPR', 'ICCV', 'ECCV'],  # 关注哪些会议
    keywords=['vision transformer', 'efficient', 'attention'],  # 关键词
    days=90,  # 最近 90 天
    max_papers=50  # 读 50 篇论文
)

# gaps: 识别的研究空白
# ideas: 基于空白生成的具体想法
```

### 方式 2: 边读边记录

```python
from cvpr_auto.paper_tracker import IdeaRecorder, PaperFetcher

# 创建记录器
recorder = IdeaRecorder()

# 边读边记录想法
recorder.record_idea(
    paper=current_paper,
    trigger_text="However, this method requires extensive computational resources",  # 哪句话触发
    idea_description="Develop a lightweight version using knowledge distillation",
    improvement="Reduce FLOPs by 50% while maintaining accuracy",
    priority='high'
)

# 读完后，从所有笔记生成结构化想法
ideas = recorder.generate_ideas_from_notes()
```

### 方式 3: 混合模式（集成到主流程）

```bash
# 新增 --paper-driven 参数
python -m cvpr_auto.main \
    --dataset imagenet \
    --paper-driven \                    # 启用文献驱动模式
    --venues CVPR ICCV ECCV \          # 调研这些会议
    --keywords "efficient architecture" \  # 关注这些方向
    --num-papers 50                    # 读 50 篇论文
```

## 技术实现

### 新增的模块

```
cvpr_auto/paper_tracker.py
├── PaperFetcher          # 抓取论文（arXiv/Semantic Scholar/CVF）
├── PaperAnalyzer         # LLM 分析论文内容
├── GapAnalyzer          # 识别研究空白
├── IdeaRecorder         # 边读边记录
└── PaperDrivenIdeaPipeline  # 完整管道
```

### 数据来源

| 来源 | 覆盖 | 更新频率 |
|------|------|---------|
| arXiv API | cs.CV 每日更新 | 实时 |
| Semantic Scholar | 全库 | 每周 |
| CVF OpenAccess | CVPR/ICCV/ECCV | 会议后 |

### LLM 分析维度

每篇论文自动提取：
- **Problem Statement**: 解决什么问题
- **Method Summary**: 方法概述
- **Key Contributions**: 主要贡献
- **Limitations**: 局限性和未解决问题 ⭐（用于识别 gaps）

## 示例输出

### 研究空白识别

```json
{
  "gaps": [
    {
      "description": "Vision Transformers are computationally expensive for mobile deployment. Most efficient variants sacrifice too much accuracy.",
      "supporting_papers": ["arxiv_2401.12345", "s2_abc123"],
      "potential_approach": "Develop a hybrid CNN-Transformer architecture with dynamic attention switching",
      "impact_score": 8.5,
      "feasibility_score": 7.0
    },
    {
      "description": "Current diffusion models require 100+ inference steps. Few works explore step reduction without quality loss.",
      "supporting_papers": ["arxiv_2402.56789"],
      "potential_approach": "Use knowledge distillation from 100-step to 10-step model with consistency regularization",
      "impact_score": 9.0,
      "feasibility_score": 6.5
    }
  ]
}
```

### 生成的想法

```json
{
  "ideas": [
    {
      "title": "MobileFormer: Efficient Vision Transformers via Dynamic Attention Switching",
      "problem": "Vision Transformers achieve high accuracy but are too slow for mobile devices. Existing efficient variants significantly degrade performance.",
      "solution": "Propose a hybrid architecture that dynamically switches between CNN and Transformer blocks based on input complexity. Simple images use CNN, complex images activate Transformer.",
      "contributions": [
        "Novel dynamic switching mechanism with negligible overhead",
        "Mobile-friendly architecture achieving 80% of ViT accuracy with 20% computation",
        "Extensive experiments on ImageNet and COCO"
      ],
      "feasibility": "high"
    }
  ]
}
```

## 优势对比

| 维度 | 原版（随机生成） | 增强版（文献驱动） |
|------|----------------|------------------|
| **新颖性** | ❌ 可能重复已有工作 | ✅ 基于真实 gaps |
| **时效性** | ❌ 不了解最新趋势 | ✅ 读最新 3 个月论文 |
| **针对性** | ❌ 想法较泛 | ✅ 针对具体 unsolved problems |
| **成功率** | ~5% | ~15-20%（估计） |
| **时间成本** | 10 分钟 | 1-2 小时（读论文） |

## 最佳实践

### 1. 定期更新文献库

```bash
# 每周运行一次，更新论文库
python -c "
from cvpr_auto.paper_tracker import PaperFetcher
fetcher = PaperFetcher()
papers = fetcher.fetch_recent_papers(days=7, max_papers=50)
print(f'Updated with {len(papers)} new papers')
"
```

### 2. 建立个人研究笔记

```python
# 长期积累研究想法
recorder = IdeaRecorder('./my_research_notes.json')

# 每次读论文时记录
recorder.record_idea(...)

# 定期回顾和整理
ideas = recorder.generate_ideas_from_notes()
```

### 3. 结合两者使用

```
Step 1: 文献驱动生成 3-5 个想法（高质量，针对性）
Step 2: 随机生成 5-10 个想法（多样性）
Step 3: 合并评估，选择最佳想法执行
```

## 与现有流程集成

### 修改主函数

```python
def generate_ideas_phase(args) -> Dict:
    if args.paper_driven:
        # 文献驱动模式
        pipeline = PaperDrivenIdeaPipeline()
        gaps, ideas = pipeline.run_full_pipeline(
            venues=args.venues,
            keywords=args.keywords,
            max_papers=args.num_papers
        )
        return ideas[0]  # 选择最佳想法
    else:
        # 原版随机生成
        from cvpr_auto.idea_generator import IdeaSelectionPipeline
        pipeline = IdeaSelectionPipeline()
        return pipeline.generate_and_select(...)
```

## 未来改进

- [ ] 集成 PDF 全文解析（当前只用摘要）
- [ ] 引用网络分析（识别 influential papers）
- [ ] 代码可用性检查（过滤无代码的论文）
- [ ] 多模态论文支持（视频、3D 等）
- [ ] 与 OpenReview 集成（获取评审意见）

---

**总结**：文献驱动的想法生成让系统从"闭门造车"变为"站在前沿看问题"，显著提高想法的新颖性和针对性。
