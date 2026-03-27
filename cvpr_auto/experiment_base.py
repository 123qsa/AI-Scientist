"""
CVPR 专用模板基础框架
支持 ImageNet 分类、COCO 检测/分割、ADE20K 语义分割
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse


class CVPRExperimentBase:
    """CVPR 实验基类"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None

    def _setup_device(self):
        """设置计算设备 (多 GPU 支持)"""
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs")
            return torch.device("cuda")
        return torch.device("cpu")

    def setup_data(self, dataset_name: str, data_dir: str):
        """设置数据集"""
        if dataset_name == "imagenet":
            return self._setup_imagenet(data_dir)
        elif dataset_name == "cifar10":
            return self._setup_cifar10(data_dir)
        elif dataset_name == "coco":
            return self._setup_coco(data_dir)
        elif dataset_name == "ade20k":
            return self._setup_ade20k(data_dir)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def _setup_imagenet(self, data_dir: str):
        """ImageNet 数据加载"""
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        train_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'train'),
            transform=train_transform
        )
        val_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'val'),
            transform=val_transform
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 256),
            shuffle=True,
            num_workers=self.config.get('num_workers', 8),
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 256),
            shuffle=False,
            num_workers=self.config.get('num_workers', 8),
            pin_memory=True
        )

        return len(train_dataset), len(val_dataset)

    def _setup_cifar10(self, data_dir: str):
        """CIFAR-10 快速测试"""
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])

        train_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=train_transform
        )
        val_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=val_transform
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=128, shuffle=True, num_workers=4
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=128, shuffle=False, num_workers=4
        )

        return len(train_dataset), len(val_dataset)

    def _setup_coco(self, data_dir: str):
        """COCO 检测/分割 (使用 detectron2 或 mmdetection)"""
        # 需要安装 detectron2 或 mmcv
        raise NotImplementedError("COCO requires detectron2/mmdetection setup")

    def _setup_ade20k(self, data_dir: str):
        """ADE20K 语义分割"""
        # 需要安装 mmsegmentation
        raise NotImplementedError("ADE20K requires mmsegmentation setup")

    def setup_model(self, model: nn.Module):
        """设置模型 (支持多 GPU)"""
        self.model = model.to(self.device)

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

    def setup_optimizer(self, optimizer_name: str, **kwargs):
        """设置优化器"""
        if optimizer_name.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=kwargs.get('lr', 1e-3),
                weight_decay=kwargs.get('weight_decay', 0.05)
            )
        elif optimizer_name.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=kwargs.get('lr', 0.1),
                momentum=kwargs.get('momentum', 0.9),
                weight_decay=kwargs.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def setup_scheduler(self, scheduler_name: str, **kwargs):
        """设置学习率调度器"""
        if scheduler_name.lower() == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=kwargs.get('epochs', 100)
            )
        elif scheduler_name.lower() == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )

    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(self.train_loader)

        return {
            'train_loss': avg_loss,
            'train_acc': accuracy
        }

    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        top5_correct = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = nn.CrossEntropyLoss()(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # Top-5 accuracy
                _, pred5 = output.topk(5, 1, True, True)
                pred5 = pred5.t()
                correct5 = pred5.eq(target.view(1, -1).expand_as(pred5))
                top5_correct += correct5[:5].reshape(-1).float().sum(0).item()

        accuracy = 100. * correct / total
        top5_acc = 100. * top5_correct / total
        avg_loss = total_loss / len(self.val_loader)

        return {
            'val_loss': avg_loss,
            'val_acc': accuracy,
            'val_top5_acc': top5_acc
        }

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CVPR Experiment Base')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['imagenet', 'cifar10', 'coco', 'ade20k'])
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adamw', 'sgd'])
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step'])
    parser.add_argument('--out-dir', type=str, default='run_0')
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # 示例使用
    config = {
        'batch_size': args.batch_size,
        'num_workers': 8
    }

    experiment = CVPRExperimentBase(config)
    train_size, val_size = experiment.setup_data(args.dataset, args.data_dir)
    print(f"Dataset: {args.dataset}, Train: {train_size}, Val: {val_size}")
