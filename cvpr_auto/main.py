#!/usr/bin/env python3
"""
CVPR-Auto: 全自动 CVPR 级别论文生成系统主入口

Usage:
    python cvpr_auto/main.py --dataset imagenet --task classification
    python cvpr_auto/main.py --dataset coco --task detection
    python cvpr_auto/main.py --quick-test  # 使用 CIFAR-10 快速测试
"""

import argparse
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from cvpr_auto.config import config
from cvpr_auto.remote_runner import check_server_connection, ensure_project_on_server


def parse_args():
    parser = argparse.ArgumentParser(
        description='CVPR-Auto: Fully Automated CVPR Paper Generation'
    )

    # 实验配置
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        choices=['imagenet', 'coco', 'ade20k', 'cifar10'],
        help='Dataset for experiments'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='classification',
        choices=['classification', 'detection', 'segmentation'],
        help='Computer vision task'
    )
    parser.add_argument(
        '--num-ideas',
        type=int,
        default=5,
        help='Number of ideas to generate and evaluate'
    )
    parser.add_argument(
        '--max-revision-rounds',
        type=int,
        default=5,
        help='Maximum self-review and revision rounds'
    )

    # 服务器配置
    parser.add_argument(
        '--local',
        action='store_true',
        help='Force local execution (not recommended for CVPR scale)'
    )

    # 快速测试
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test with CIFAR-10 (for debugging)'
    )

    # 输出配置
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./cvpr_outputs',
        help='Output directory for paper and results'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("🚀 CVPR-Auto: Fully Automated CVPR Paper Generation System")
    print("=" * 70)

    # 快速测试模式
    if args.quick_test:
        print("\n⚡ Quick Test Mode (CIFAR-10)")
        args.dataset = 'cifar10'
        args.num_ideas = 2
        config.QUALITY_THRESHOLDS['min_datasets'] = 1
        config.QUALITY_THRESHOLDS['min_ablations'] = 3

    # 检查服务器连接
    if not args.local:
        print("\n☁️  Checking cloud server connection...")
        if not check_server_connection():
            print("❌ Cannot connect to server. Options:")
            print("   1. Check VPN/SSH configuration")
            print("   2. Use --local for local execution (not for CVPR scale)")
            sys.exit(1)

        print(f"✅ Connected to {config.SERVER_IP}")
        ensure_project_on_server()

    # 显示配置
    print(f"\n📋 Configuration:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Task: {args.task}")
    print(f"   Ideas to generate: {args.num_ideas}")
    print(f"   Max revision rounds: {args.max_revision_rounds}")
    print(f"   Execution mode: {'Local' if args.local else 'Cloud'}")
    print(f"   Output: {args.output_dir}")

    print("\n" + "=" * 70)
    print("🎯 Phase 1: Idea Generation & Novelty Verification")
    print("=" * 70)
    # TODO: 调用 idea_generator 模块
    print("(Generating and validating novel ideas...)")

    print("\n" + "=" * 70)
    print("🔬 Phase 2: Large-Scale Experimentation")
    print("=" * 70)
    # TODO: 调用 experiment_engine 模块
    print("(Running experiments with hyperparameter search...)")
    print("(Performing comprehensive ablation studies...)")
    print("(Comparing with SOTA methods...)")

    print("\n" + "=" * 70)
    print("📝 Phase 3: Professional Paper Composition")
    print("=" * 70)
    # TODO: 调用 paper_composer 模块
    print("(Generating CVPR-format LaTeX with professional figures...)")

    print("\n" + "=" * 70)
    print("🔍 Phase 4: Self-Review & Iterative Improvement")
    print("=" * 70)
    # TODO: 调用 self_review 模块
    print("(Simulating reviewer perspective...)")
    print("(Iterating to improve quality...)")

    print("\n" + "=" * 70)
    print("✅ Phase 5: Quality Assessment & Output")
    print("=" * 70)
    # TODO: 质量评估
    print("(Final quality check against CVPR standards...)")

    print("\n" + "=" * 70)
    print("🎉 CVPR Paper Generation Complete!")
    print("=" * 70)
    print(f"\nOutput Location: {args.output_dir}/")
    print("\nFiles generated:")
    print("  - paper.pdf         (Main paper)")
    print("  - supplementary.pdf (Supplementary material)")
    print("  - code/             (Reproducible code)")
    print("  - review_report.json (Self-review report)")
    print("\nNext steps:")
    print("  1. Review the generated paper")
    print("  2. Check experiment reproducibility")
    print("  3. Consider professional proofreading")
    print("  4. Submit to ArXiv for community feedback")
    print("  5. Submit to CVPR!")


if __name__ == '__main__':
    main()
