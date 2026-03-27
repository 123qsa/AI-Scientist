"""
远程服务器执行模块 - CVPR-Auto 专用
所有实验通过 SSH 在云端服务器运行
"""

import subprocess
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 从配置文件导入
from cvpr_auto.config import config


def get_ssh_base_cmd():
    """获取基础 SSH 命令"""
    return [
        "ssh",
        "-i", config.SSH_KEY,
        "-p", str(config.SERVER_PORT),
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        f"{config.SERVER_USER}@{config.SERVER_IP}"
    ]


def check_server_connection() -> bool:
    """检查服务器连接"""
    try:
        result = subprocess.run(
            get_ssh_base_cmd() + ["echo 'connected'"],
            capture_output=True,
            text=True,
            timeout=15
        )
        return result.returncode == 0 and "connected" in result.stdout
    except Exception as e:
        print(f"❌ 无法连接服务器: {e}")
        return False


def ensure_project_on_server() -> bool:
    """确保项目已部署到服务器"""
    print("📡 检查服务器项目...")

    remote_dir = "~/AI-Scientist"

    # 检查目录是否存在
    check_cmd = get_ssh_base_cmd() + [f"test -d {remote_dir} && echo 'exists'"]
    result = subprocess.run(check_cmd, capture_output=True, text=True)

    if "exists" not in result.stdout:
        print("📥 项目不存在，正在克隆...")
        clone_cmd = get_ssh_base_cmd() + [
            f"git clone https://github.com/123qsa/AI-Scientist.git {remote_dir}"
        ]
        subprocess.run(clone_cmd, check=True)
        print("✅ 项目克隆完成")

        # 安装依赖
        print("📦 安装依赖...")
        install_cmd = get_ssh_base_cmd() + [
            f"cd {remote_dir} && pip install -r requirements.txt"
        ]
        subprocess.run(install_cmd, check=True)
    else:
        # 更新代码
        print("🔄 更新项目代码...")
        pull_cmd = get_ssh_base_cmd() + [
            f"cd {remote_dir} && git pull"
        ]
        subprocess.run(pull_cmd, check=True)

    return True


def run_cvpr_experiment_on_server(
    dataset: str = "imagenet",
    task: str = "classification",
    num_ideas: int = 5,
    max_revision_rounds: int = 5,
    **kwargs
) -> int:
    """在服务器上运行 CVPR 实验"""

    if not check_server_connection():
        sys.exit(1)

    ensure_project_on_server()

    remote_dir = "~/AI-Scientist"

    # 构建命令
    cmd_parts = [
        "cd", remote_dir,
        "&&", "python", "-m", "cvpr_auto.main",
        "--dataset", dataset,
        "--task", task,
        "--num-ideas", str(num_ideas),
        "--max-revision-rounds", str(max_revision_rounds),
    ]

    for key, value in kwargs.items():
        if value is True:
            cmd_parts.append(f"--{key.replace('_', '-')}")
        elif value and not isinstance(value, bool):
            cmd_parts.extend([f"--{key.replace('_', '-')}", str(value)])

    # 环境变量
    env_setup = ""
    if "OPENALEX_MAIL_ADDRESS" in os.environ:
        env_setup = f"export OPENALEX_MAIL_ADDRESS='{os.environ['OPENALEX_MAIL_ADDRESS']}' && "

    full_cmd = " ".join(cmd_parts)
    ssh_cmd = get_ssh_base_cmd() + [f"{env_setup}{full_cmd}"]

    print(f"\n🚀 在服务器上启动 CVPR 实验...")
    print(f"   数据集: {dataset}")
    print(f"   任务: {task}")
    print(f"   想法数: {num_ideas}")
    print(f"\n{'='*60}")

    # 实时输出
    process = subprocess.Popen(
        ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end='')

    process.wait()

    if process.returncode == 0:
        print(f"\n{'='*60}")
        print("✅ 实验完成！")
        # 同步结果到本地
        sync_results_from_server()
    else:
        print(f"\n{'='*60}")
        print("❌ 实验失败！")

    return process.returncode


def sync_results_from_server():
    """同步结果到本地"""
    print("\n📥 同步结果到本地...")

    local_results = Path("cvpr_outputs")
    local_results.mkdir(exist_ok=True)

    remote_dir = "~/AI-Scientist"

    rsync_cmd = [
        "rsync",
        "-avz",
        "-e", f"ssh -i {config.SSH_KEY} -p {config.SERVER_PORT}",
        f"{config.SERVER_USER}@{config.SERVER_IP}:{remote_dir}/cvpr_outputs/",
        str(local_results) + "/"
    ]

    try:
        subprocess.run(rsync_cmd, check=True)
        print("✅ 结果同步完成")
    except subprocess.CalledProcessError:
        print("⚠️  rsync 失败，尝试使用 scp...")
        scp_cmd = [
            "scp", "-i", config.SSH_KEY, "-P", str(config.SERVER_PORT), "-r",
            f"{config.SERVER_USER}@{config.SERVER_IP}:{remote_dir}/cvpr_outputs/*",
            str(local_results) + "/"
        ]
        subprocess.run(scp_cmd, capture_output=True)


if __name__ == "__main__":
    # 测试连接
    if check_server_connection():
        print("✅ 服务器连接正常")
        ensure_project_on_server()
    else:
        print("❌ 服务器连接失败")
        sys.exit(1)
