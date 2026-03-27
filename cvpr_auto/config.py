"""
CVPR-Auto 配置文件
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class CVPRConfig:
    """CVPR 级别论文生成配置"""

    # === 服务器配置 ===
    SERVER_IP: str = "166.111.86.21"
    SERVER_USER: str = "hanjiajun"
    SERVER_PORT: int = 22
    SSH_KEY: str = field(default_factory=lambda: os.path.expanduser(
        "~/Desktop/服务器公私钥/id_ed25526574_qq_com"
    ))

    # === 实验配置 ===
    # 支持的大规模数据集
    DATASETS: Dict[str, Dict] = field(default_factory=lambda: {
        "imagenet": {
            "path": "/data/imagenet",
            "num_classes": 1000,
            "split": ["train", "val"],
            "metrics": ["top1", "top5"]
        },
        "coco": {
            "path": "/data/coco",
            "tasks": ["detection", "instance_seg", "keypoint"],
            "metrics": ["AP", "AP50", "AP75"]
        },
        "ade20k": {
            "path": "/data/ade20k",
            "task": "semantic_segmentation",
            "metrics": ["mIoU", "aAcc", "Acc"]
        },
        "cifar10": {
            "path": "/data/cifar10",
            "num_classes": 10,
            "quick_test": True
        }
    })

    # GPU 配置
    GPUS: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    DISTRIBUTED: bool = True
    NUM_WORKERS: int = 8

    # === 实验搜索空间 ===
    HYPERPARAM_SEARCH: Dict = field(default_factory=lambda: {
        "learning_rate": {"type": "log", "low": 1e-5, "high": 1e-2},
        "batch_size": {"type": "choice", "options": [64, 128, 256, 512]},
        "optimizer": {"type": "choice", "options": ["adamw", "sgd", "lars"]},
        "scheduler": {"type": "choice", "options": ["cosine", "step", "plateau"]},
        "weight_decay": {"type": "log", "low": 1e-6, "high": 1e-3},
        "epochs": {"type": "choice", "options": [100, 200, 300]},
        "warmup_epochs": {"type": "int", "low": 5, "high": 20}
    })

    # === 论文质量阈值 (CVPR 标准) ===
    QUALITY_THRESHOLDS: Dict = field(default_factory=lambda: {
        "novelty_score": 7.5,          # 创新性 (1-10)
        "experiment_rigor": 8.0,        # 实验严谨性 (1-10)
        "writing_quality": 7.5,         # 写作质量 (1-10)
        "significance": 7.0,            # 重要性 (1-10)
        "min_improvement": 1.0,         # 相比 SOTA 最小提升 (%)
        "min_ablations": 5,             # 最少消融实验数
        "min_datasets": 2,              # 最少数据集数
        "min_baselines": 5              # 最少对比 baseline 数
    })

    # === 写作配置 ===
    PAPER_SECTIONS: List[str] = field(default_factory=lambda: [
        "abstract",
        "introduction",
        "related_work",
        "method",
        "experiments",
        "conclusion",
        "supplementary"
    ])

    MAX_REVISION_ROUNDS: int = 5
    TARGET_PAGE_LENGTH: int = 8  # CVPR 主会要求

    # === API 配置 ===
    LLM_MODEL: str = "kimi-k2.5"  # 或 "claude-3-5-sonnet", "gpt-4o"
    LLM_TEMPERATURE: float = 0.7

    # === 输出配置 ===
    OUTPUT_DIR: str = "./cvpr_outputs"
    SAVE_CHECKPOINTS: bool = True
    GENERATE_SUPP: bool = True


# 全局配置实例
config = CVPRConfig()
