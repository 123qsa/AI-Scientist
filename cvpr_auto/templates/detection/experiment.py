"""
COCO 检测任务模板
基于 Detectron2
"""

import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode

import os
import json
import numpy as np
from pathlib import Path


class COCODetectionExperiment:
    """COCO 检测实验"""

    def __init__(self, config: dict):
        self.config = config
        self.cfg = get_cfg()
        self.output_dir = Path(config.get('out_dir', 'run_0'))
        self.output_dir.mkdir(exist_ok=True)

    def setup_config(self):
        """配置 Detectron2"""
        # 使用预训练模型
        model_name = self.config.get('model', 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml')
        self.cfg.merge_from_file(model_zoo.get_config_file(model_name))

        # 数据集配置
        self.cfg.DATASETS.TRAIN = ("coco_2017_train",)
        self.cfg.DATASETS.TEST = ("coco_2017_val",)

        # 数据加载
        self.cfg.DATALOADER.NUM_WORKERS = self.config.get('num_workers', 4)

        # 模型配置
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80

        # 训练配置
        self.cfg.SOLVER.IMS_PER_BATCH = self.config.get('batch_size', 4)
        self.cfg.SOLVER.BASE_LR = self.config.get('lr', 0.00025)
        self.cfg.SOLVER.MAX_ITER = self.config.get('max_iter', 10000)
        self.cfg.SOLVER.STEPS = []  # 学习率衰减步骤

        self.cfg.OUTPUT_DIR = str(self.output_dir)

    def train(self):
        """训练模型"""
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    def evaluate(self):
        """评估模型"""
        evaluator = COCOEvaluator("coco_2017_val", self.cfg, False, output_dir=str(self.output_dir))
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=True)

        from detectron2.evaluation import inference_on_dataset
        from detectron2.data import build_detection_test_loader

        val_loader = build_detection_test_loader(self.cfg, "coco_2017_val")
        results = inference_on_dataset(trainer.model, val_loader, evaluator)

        # 保存结果
        with open(self.output_dir / 'coco_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        return results


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='run_0')
    parser.add_argument('--max_iter', type=int, default=10000)
    args = parser.parse_args()

    config = {
        'out_dir': args.out_dir,
        'max_iter': args.max_iter,
        'batch_size': 4,
        'lr': 0.00025
    }

    exp = COCODetectionExperiment(config)
    exp.setup_config()
    exp.train()
    results = exp.evaluate()

    print(f"Results: {results}")


if __name__ == '__main__':
    main()
