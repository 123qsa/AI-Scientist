#!/usr/bin/env python3
"""
AI-Scientist 远程执行入口
默认通过 SSH 在云端服务器运行所有实验
本地仅作为控制台显示输出

使用方法:
    ./launch_scientist_remote.py [参数与原版相同]

环境变量:
    LOCAL_MODE=1  # 强制本地运行（调试用）
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from remote_runner import (
    run_experiment_on_server,
    prepare_baseline_on_server,
    check_baseline_on_server,
    check_server_connection
)


def parse_remote_args(args):
    """解析参数"""
    import argparse

    parser = argparse.ArgumentParser(description="Run AI Scientist on remote server")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Force local execution (debug only)"
    )
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="Skip idea generation"
    )
    parser.add_argument(
        "--skip-novelty-check",
        action="store_true",
        help="Skip novelty check"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="nanoGPT_lite",
        help="Experiment to run"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="kimi-k2.5",
        help="Model to use"
    )
    parser.add_argument(
        "--num-ideas",
        type=int,
        default=2,
        help="Number of ideas to generate"
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="openalex",
        choices=["semanticscholar", "openalex"],
        help="Scholar engine"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Number of parallel processes"
    )
    parser.add_argument(
        "--improvement",
        action="store_true",
        help="Improve based on reviews"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs"
    )
    parser.add_argument(
        "--prepare-baseline",
        action="store_true",
        help="Prepare baseline on server"
    )
    parser.add_argument(
        "--check-baseline",
        action="store_true",
        help="Check baseline status on server"
    )

    return parser.parse_args(args)


def main():
    """主函数"""
    args = parse_remote_args(sys.argv[1:])

    # 检查是否强制本地运行
    if args.local or os.environ.get("LOCAL_MODE") == "1":
        print("🖥️  本地模式（调试）")
        # 导入并运行原版
        import launch_scientist
        launch_scientist.main()
        return

    # 远程模式
    print("☁️  云端模式 - 实验将在远程服务器运行\n")

    # 检查服务器
    if not check_server_connection():
        print("\n❌ 无法连接服务器，请检查：")
        print(f"   1. VPN/网络连接")
        print(f"   2. SSH 密钥: {os.path.expanduser('~/Desktop/服务器公私钥/id_ed25526574_qq_com')}")
        print(f"\n或使用本地模式: LOCAL_MODE=1 python {sys.argv[0]}")
        sys.exit(1)

    # 准备 baseline
    if args.prepare_baseline:
        success = prepare_baseline_on_server(args.experiment)
        sys.exit(0 if success else 1)

    # 检查 baseline
    if args.check_baseline:
        status = check_baseline_on_server(args.experiment)
        print(status)
        sys.exit(0)

    # 运行实验
    kwargs = {
        "skip_idea_generation": args.skip_idea_generation,
        "skip_novelty_check": args.skip_novelty_check,
        "improvement": args.improvement,
    }
    if args.gpus:
        kwargs["gpus"] = args.gpus

    exit_code = run_experiment_on_server(
        model=args.model,
        experiment=args.experiment,
        num_ideas=args.num_ideas,
        engine=args.engine,
        parallel=args.parallel,
        **kwargs
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
