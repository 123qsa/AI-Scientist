#!/usr/bin/env python3
"""
CVPR-Auto 远程部署 CLI
一键部署和监控远程实验
"""

import argparse
import json
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from cvpr_auto.remote_manager import (
    ServerRegistry, ServerConfig, RemoteServerManager, deploy_to_server
)


def cmd_add_server(args):
    """添加服务器配置"""
    registry = ServerRegistry()

    config = ServerConfig(
        host=args.host,
        user=args.user,
        port=args.port,
        key_file=args.key_file,
        workspace=args.workspace,
        python_env=args.venv,
    )

    registry.add(args.name, config)
    print(f"✅ 已添加服务器: {args.name}")
    print(f"   主机: {args.host}")
    print(f"   用户: {args.user}")
    print(f"   工作目录: {args.workspace}")


def cmd_list_servers(args):
    """列出服务器"""
    registry = ServerRegistry()
    servers = registry.list_servers()

    if not servers:
        print("未配置任何服务器")
        return

    print(f"\n{'='*70}")
    print(f"已配置的服务器 ({len(servers)}个):")
    print(f"{'='*70}")

    for name in servers:
        cfg = registry.get(name)
        print(f"\n📌 {name}")
        print(f"   主机: {cfg.host}")
        print(f"   用户: {cfg.user}")
        print(f"   端口: {cfg.port}")
        print(f"   工作目录: {cfg.workspace}")
        print(f"   Python环境: {cfg.python_env}")


def cmd_remove_server(args):
    """删除服务器"""
    registry = ServerRegistry()
    registry.remove(args.name)
    print(f"✅ 已删除服务器: {args.name}")


def cmd_check(args):
    """检查服务器连接"""
    registry = ServerRegistry()
    config = registry.get(args.name)

    if not config:
        print(f"❌ 未找到服务器: {args.name}")
        return

    print(f"正在检查 {args.name} 连接...")
    manager = RemoteServerManager(config)

    if manager.check_connection():
        print("✅ SSH 连接成功")

        # 检查 GPU
        code, stdout, _ = manager.run_remote("nvidia-smi --query-gpu=name --format=csv,noheader")
        if code == 0:
            gpus = [g.strip() for g in stdout.strip().split('\n') if g.strip()]
            print(f"✅ 检测到 {len(gpus)} 个 GPU:")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
        else:
            print("⚠️ 未检测到 GPU")

        # 检查目录
        code, _, _ = manager.run_remote(f"ls {config.workspace}")
        if code == 0:
            print(f"✅ 工作目录存在: {config.workspace}")
        else:
            print(f"⚠️ 工作目录不存在，将自动创建")
    else:
        print("❌ SSH 连接失败")
        print("请检查:")
        print("  1. 服务器是否可达")
        print("  2. SSH 密钥是否正确配置")
        print("  3. 用户名和主机地址是否正确")


def cmd_sync(args):
    """同步代码"""
    registry = ServerRegistry()
    config = registry.get(args.name)

    if not config:
        print(f"❌ 未找到服务器: {args.name}")
        return

    print(f"正在同步代码到 {args.name}...")
    manager = RemoteServerManager(config)

    if manager.sync_code(args.local_dir):
        print("✅ 代码同步完成")
    else:
        print("❌ 代码同步失败")


def cmd_deploy(args):
    """部署并运行实验"""
    print(f"🚀 部署实验到 {args.server}...")
    print(f"   模板: {args.template}")
    print(f"   模型: {args.model}")
    print(f"   CVPR模式: {args.cvpr_mode}")

    success = deploy_to_server(
        server_name=args.server,
        template=args.template,
        model=args.model,
        num_ideas=args.num_ideas,
        cvpr_mode=args.cvpr_mode,
        max_iterations=args.max_iterations,
        quality_threshold=args.quality_threshold,
        parallel=args.parallel,
        improvement=args.improvement,
    )

    if success:
        print("\n✅ 实验已启动!")
        print(f"使用 'python -m cvpr_auto.remote_cli status {args.server}' 查看状态")
    else:
        print("\n❌ 部署失败")


def cmd_status(args):
    """查看实验状态"""
    registry = ServerRegistry()
    config = registry.get(args.name)

    if not config:
        print(f"❌ 未找到服务器: {args.name}")
        return

    manager = RemoteServerManager(config)
    status = manager.get_experiment_status()

    print(f"\n{'='*70}")
    print(f"📊 服务器状态: {args.name}")
    print(f"{'='*70}")

    if status["is_running"]:
        print("\n🟢 实验运行中")
        print("\n进程信息:")
        print(status["processes"])
    else:
        print("\n⚪ 没有运行中的实验")

    print("\n🎮 GPU 状态:")
    print(status["gpu_status"])

    print("\n📝 最近日志:")
    print("-" * 50)
    print(status["recent_logs"][-500:] if len(status["recent_logs"]) > 500 else status["recent_logs"])


def cmd_logs(args):
    """查看完整日志"""
    registry = ServerRegistry()
    config = registry.get(args.name)

    if not config:
        print(f"❌ 未找到服务器: {args.name}")
        return

    manager = RemoteServerManager(config)
    code, stdout, _ = manager.run_remote(f"cat {config.workspace}/experiment.log")

    if code == 0:
        print(stdout)
    else:
        print("无法读取日志文件")


def cmd_kill(args):
    """终止实验"""
    registry = ServerRegistry()
    config = registry.get(args.name)

    if not config:
        print(f"❌ 未找到服务器: {args.name}")
        return

    manager = RemoteServerManager(config)
    if manager.kill_experiment():
        print(f"✅ 已终止 {args.name} 上的实验")
    else:
        print(f"⚠️ {args.name} 上没有运行中的实验")


def cmd_download(args):
    """下载结果"""
    registry = ServerRegistry()
    config = registry.get(args.name)

    if not config:
        print(f"❌ 未找到服务器: {args.name}")
        return

    manager = RemoteServerManager(config)
    if manager.download_results(args.output_dir):
        print(f"✅ 结果已下载到 {args.output_dir}")
    else:
        print("❌ 下载失败")


def main():
    parser = argparse.ArgumentParser(
        description="CVPR-Auto 远程部署工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 添加服务器
  python -m cvpr_auto.remote_cli add myserver --host 192.168.1.100 --user ubuntu --key-file ~/.ssh/id_rsa

  # 检查连接
  python -m cvpr_auto.remote_cli check myserver

  # 部署实验
  python -m cvpr_auto.remote_cli deploy myserver --template cvpr_lite --model kimi-k2.5

  # 查看状态
  python -m cvpr_auto.remote_cli status myserver

  # 下载结果
  python -m cvpr_auto.remote_cli download myserver --output ./my_results
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # add
    add_parser = subparsers.add_parser("add", help="添加服务器配置")
    add_parser.add_argument("name", help="服务器名称")
    add_parser.add_argument("--host", required=True, help="主机地址")
    add_parser.add_argument("--user", required=True, help="用户名")
    add_parser.add_argument("--port", type=int, default=22, help="SSH端口")
    add_parser.add_argument("--key-file", help="SSH私钥文件")
    add_parser.add_argument("--workspace", default="/workspace/AI-Scientist", help="远程工作目录")
    add_parser.add_argument("--venv", default="venv_4090", help="Python虚拟环境名称")
    add_parser.set_defaults(func=cmd_add_server)

    # list
    list_parser = subparsers.add_parser("list", help="列出服务器")
    list_parser.set_defaults(func=cmd_list_servers)

    # remove
    remove_parser = subparsers.add_parser("remove", help="删除服务器配置")
    remove_parser.add_argument("name", help="服务器名称")
    remove_parser.set_defaults(func=cmd_remove_server)

    # check
    check_parser = subparsers.add_parser("check", help="检查服务器连接")
    check_parser.add_argument("name", help="服务器名称")
    check_parser.set_defaults(func=cmd_check)

    # sync
    sync_parser = subparsers.add_parser("sync", help="同步代码")
    sync_parser.add_argument("name", help="服务器名称")
    sync_parser.add_argument("--local-dir", default=".", help="本地代码目录")
    sync_parser.set_defaults(func=cmd_sync)

    # deploy
    deploy_parser = subparsers.add_parser("deploy", help="部署并运行实验")
    deploy_parser.add_argument("server", help="服务器名称")
    deploy_parser.add_argument("--template", default="cvpr_lite", help="实验模板")
    deploy_parser.add_argument("--model", default="kimi-k2.5", help="LLM模型")
    deploy_parser.add_argument("--num-ideas", type=int, default=3, help="想法数量")
    deploy_parser.add_argument("--cvpr-mode", action="store_true", default=True, help="启用CVPR模式")
    deploy_parser.add_argument("--max-iterations", type=int, default=5, help="最大迭代次数 (1-20)")
    deploy_parser.add_argument("--quality-threshold", type=float, default=7.5, help="质量阈值")
    deploy_parser.add_argument("--parallel", type=int, default=0, help="并行进程数")
    deploy_parser.add_argument("--no-improvement", action="store_true", help="禁用论文改进")
    deploy_parser.set_defaults(func=cmd_deploy, improvement=True)

    # status
    status_parser = subparsers.add_parser("status", help="查看实验状态")
    status_parser.add_argument("name", help="服务器名称")
    status_parser.set_defaults(func=cmd_status)

    # logs
    logs_parser = subparsers.add_parser("logs", help="查看完整日志")
    logs_parser.add_argument("name", help="服务器名称")
    logs_parser.set_defaults(func=cmd_logs)

    # kill
    kill_parser = subparsers.add_parser("kill", help="终止实验")
    kill_parser.add_argument("name", help="服务器名称")
    kill_parser.set_defaults(func=cmd_kill)

    # download
    download_parser = subparsers.add_parser("download", help="下载实验结果")
    download_parser.add_argument("name", help="服务器名称")
    download_parser.add_argument("--output-dir", default="./remote_results", help="输出目录")
    download_parser.set_defaults(func=cmd_download)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 处理 improvement 参数
    if hasattr(args, 'no_improvement'):
        args.improvement = not args.no_improvement

    args.func(args)


if __name__ == "__main__":
    main()
