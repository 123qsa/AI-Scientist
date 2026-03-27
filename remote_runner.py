"""
远程服务器执行模块
所有实验通过 SSH 在云端服务器运行
"""

import subprocess
import os
import sys
from pathlib import Path

# 服务器配置（写死）
SERVER_IP = "166.111.86.21"
SERVER_USER = "hanjiajun"
SERVER_PORT = "22"
SSH_KEY = os.path.expanduser("~/Desktop/服务器公私钥/id_ed25526574_qq_com")
REMOTE_PROJECT_DIR = "~/AI-Scientist"


def get_ssh_base_cmd():
    """获取基础 SSH 命令"""
    return [
        "ssh",
        "-i", SSH_KEY,
        "-p", SERVER_PORT,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        f"{SERVER_USER}@{SERVER_IP}"
    ]


def check_server_connection():
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


def ensure_project_on_server():
    """确保项目已部署到服务器"""
    print("📡 检查服务器项目...")

    # 检查目录是否存在
    check_cmd = get_ssh_base_cmd() + [f"test -d {REMOTE_PROJECT_DIR} && echo 'exists'"]
    result = subprocess.run(check_cmd, capture_output=True, text=True)

    if "exists" not in result.stdout:
        print("📥 项目不存在，正在克隆...")
        clone_cmd = get_ssh_base_cmd() + [
            f"git clone https://github.com/123qsa/AI-Scientist.git {REMOTE_PROJECT_DIR}"
        ]
        subprocess.run(clone_cmd, check=True)
        print("✅ 项目克隆完成")

        # 安装依赖
        print("📦 安装依赖...")
        install_cmd = get_ssh_base_cmd() + [
            f"cd {REMOTE_PROJECT_DIR} && pip install -r requirements.txt"
        ]
        subprocess.run(install_cmd, check=True)
    else:
        # 更新代码
        print("🔄 更新项目代码...")
        pull_cmd = get_ssh_base_cmd() + [
            f"cd {REMOTE_PROJECT_DIR} && git pull"
        ]
        subprocess.run(pull_cmd, check=True)

    return True


def run_experiment_on_server(
    model="kimi-k2.5",
    experiment="nanoGPT_lite",
    num_ideas=2,
    engine="openalex",
    parallel=0,
    **kwargs
):
    """在服务器上运行实验"""

    if not check_server_connection():
        sys.exit(1)

    ensure_project_on_server()

    # 构建命令
    cmd_parts = [
        "cd", REMOTE_PROJECT_DIR,
        "&&", "python", "launch_scientist.py",
        "--model", model,
        "--experiment", experiment,
        "--num-ideas", str(num_ideas),
        "--engine", engine,
    ]

    if parallel > 0:
        cmd_parts.extend(["--parallel", str(parallel)])

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

    print(f"\n🚀 在服务器上启动实验...")
    print(f"   模型: {model}")
    print(f"   模板: {experiment}")
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
        sync_results()
    else:
        print(f"\n{'='*60}")
        print("❌ 实验失败！")

    return process.returncode


def sync_results():
    """同步结果到本地"""
    print("\n📥 同步结果到本地...")

    local_results = Path("results")
    local_results.mkdir(exist_ok=True)

    rsync_cmd = [
        "rsync",
        "-avz",
        "-e", f"ssh -i {SSH_KEY} -p {SERVER_PORT}",
        f"{SERVER_USER}@{SERVER_IP}:{REMOTE_PROJECT_DIR}/results/",
        str(local_results) + "/"
    ]

    try:
        subprocess.run(rsync_cmd, check=True)
        print("✅ 结果同步完成")
    except subprocess.CalledProcessError:
        print("⚠️  rsync 失败，尝试使用 scp...")
        scp_cmd = [
            "scp", "-i", SSH_KEY, "-P", SERVER_PORT, "-r",
            f"{SERVER_USER}@{SERVER_IP}:{REMOTE_PROJECT_DIR}/results/*",
            str(local_results) + "/"
        ]
        subprocess.run(scp_cmd, capture_output=True)


def prepare_baseline_on_server(experiment):
    """在服务器上准备 baseline"""
    if not check_server_connection():
        return False

    ensure_project_on_server()

    print(f"📡 在服务器上准备 {experiment} baseline...")

    cmd = get_ssh_base_cmd() + [
        f"cd {REMOTE_PROJECT_DIR}/templates/{experiment} && "
        f"python experiment.py --out_dir run_0"
    ]

    result = subprocess.run(cmd)
    return result.returncode == 0


def check_baseline_on_server(experiment):
    """检查服务器上的 baseline"""
    if not check_server_connection():
        return "❌ 无法连接服务器"

    cmd = get_ssh_base_cmd() + [
        f"test -f {REMOTE_PROJECT_DIR}/templates/{experiment}/run_0/final_info.json "
        f"&& echo 'ready' || echo 'missing'"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if "ready" in result.stdout:
        return f"✅ {experiment} baseline 已准备（服务器）"
    else:
        return f"⚠️ {experiment} 缺少 baseline（服务器）"


if __name__ == "__main__":
    # 测试连接
    if check_server_connection():
        print("✅ 服务器连接正常")
        ensure_project_on_server()
    else:
        print("❌ 服务器连接失败")
        sys.exit(1)
