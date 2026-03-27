"""
远程服务器管理模块
支持 SSH 连接、代码同步、远程实验执行
"""

import os
import json
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """服务器配置"""
    host: str
    user: str
    port: int = 22
    key_file: Optional[str] = None
    password: Optional[str] = None
    workspace: str = "/workspace/AI-Scientist"
    python_env: str = "venv_4090"

    def to_dict(self) -> Dict:
        return {
            "host": self.host,
            "user": self.user,
            "port": self.port,
            "key_file": self.key_file,
            "workspace": self.workspace,
            "python_env": self.python_env,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ServerConfig":
        return cls(
            host=data["host"],
            user=data["user"],
            port=data.get("port", 22),
            key_file=data.get("key_file"),
            workspace=data.get("workspace", "/workspace/AI-Scientist"),
            python_env=data.get("python_env", "venv_4090"),
        )


class RemoteServerManager:
    """远程服务器管理器"""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.ssh_base = self._build_ssh_base()

    def _build_ssh_base(self) -> List[str]:
        """构建 SSH 基础命令"""
        cmd = ["ssh", "-p", str(self.config.port)]
        if self.config.key_file:
            cmd.extend(["-i", self.config.key_file])
        cmd.append(f"{self.config.user}@{self.config.host}")
        return cmd

    def run_remote(self, command: str, timeout: int = 300) -> tuple[int, str, str]:
        """在远程服务器执行命令"""
        ssh_cmd = self.ssh_base + [command]
        logger.info(f"远程执行: {command}")

        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Timeout"
        except Exception as e:
            return -1, "", str(e)

    def check_connection(self) -> bool:
        """检查 SSH 连接"""
        code, stdout, stderr = self.run_remote("echo 'connected'", timeout=10)
        return code == 0 and "connected" in stdout

    def sync_code(self, local_dir: str, exclude_patterns: List[str] = None) -> bool:
        """同步代码到远程服务器"""
        exclude_patterns = exclude_patterns or [
            ".git", "__pycache__", "*.pyc", ".DS_Store",
            "results", "*.pdf", "*.log", ".claude"
        ]

        # 构建 rsync 命令
        rsync_cmd = [
            "rsync", "-avz", "--progress",
            "-e", f"ssh -p {self.config.port}" + (f" -i {self.config.key_file}" if self.config.key_file else ""),
        ]

        for pattern in exclude_patterns:
            rsync_cmd.extend(["--exclude", pattern])

        rsync_cmd.extend([
            local_dir + "/",
            f"{self.config.user}@{self.config.host}:{self.config.workspace}/"
        ])

        logger.info(f"同步代码到 {self.config.host}...")
        try:
            result = subprocess.run(rsync_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ 代码同步完成")
                return True
            else:
                logger.error(f"❌ 同步失败: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ 同步错误: {e}")
            return False

    def setup_environment(self) -> bool:
        """设置远程环境"""
        logger.info("设置远程环境...")

        commands = [
            f"cd {self.config.workspace} && python3 -m venv {self.config.python_env}",
            f"cd {self.config.workspace} && source {self.config.python_env}/bin/activate && pip install -r requirements.txt",
        ]

        for cmd in commands:
            code, stdout, stderr = self.run_remote(cmd, timeout=180)
            if code != 0:
                logger.error(f"❌ 环境设置失败: {stderr}")
                return False

        logger.info("✅ 环境设置完成")
        return True

    def run_experiment(
        self,
        template: str,
        model: str,
        num_ideas: int = 3,
        cvpr_mode: bool = True,
        max_iterations: int = 2,
        **kwargs
    ) -> str:
        """在远程服务器运行实验"""
        # 构建命令
        cmd_parts = [
            f"cd {self.config.workspace}",
            f"source {self.config.python_env}/bin/activate",
            f"python launch_scientist.py",
            f"--experiment {template}",
            f"--model {model}",
            f"--num-ideas {num_ideas}",
        ]

        if cvpr_mode:
            cmd_parts.extend([
                "--cvpr-mode",
                f"--max-iterations {max_iterations}",
                f"--quality-threshold {kwargs.get('quality_threshold', 7.5)}",
            ])

        if kwargs.get("paper_driven"):
            cmd_parts.append("--paper-driven")
            if kwargs.get("keywords"):
                cmd_parts.extend(["--keywords", f"\"{kwargs['keywords']}\""])

        if kwargs.get("parallel", 0) > 0:
            cmd_parts.append(f"--parallel {kwargs['parallel']}")

        if kwargs.get("improvement", True):
            cmd_parts.append("--improvement")

        # 使用 nohup 后台运行
        full_cmd = " && ".join(cmd_parts)
        remote_cmd = f"cd {self.config.workspace} && nohup bash -c '{full_cmd}' > experiment.log 2>&1 & echo $!"

        code, stdout, stderr = self.run_remote(remote_cmd)
        if code == 0:
            pid = stdout.strip()
            logger.info(f"✅ 实验已启动，PID: {pid}")
            return pid
        else:
            logger.error(f"❌ 启动失败: {stderr}")
            return ""

    def get_experiment_status(self) -> Dict:
        """获取实验状态"""
        # 检查是否有运行中的 Python 进程
        code, stdout, stderr = self.run_remote(
            f"ps aux | grep 'launch_scientist.py' | grep -v grep || echo 'No process'"
        )

        is_running = "launch_scientist.py" in stdout

        # 获取日志
        log_cmd = f"tail -n 50 {self.config.workspace}/experiment.log 2>/dev/null || echo 'No log file'"
        _, logs, _ = self.run_remote(log_cmd)

        # 获取 GPU 状态
        gpu_cmd = "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU'"
        _, gpu_info, _ = self.run_remote(gpu_cmd)

        return {
            "is_running": is_running,
            "processes": stdout.strip() if is_running else "No running processes",
            "recent_logs": logs,
            "gpu_status": gpu_info,
        }

    def download_results(self, local_dir: str = "./remote_results") -> bool:
        """下载实验结果"""
        local_path = Path(local_dir)
        local_path.mkdir(exist_ok=True)

        rsync_cmd = [
            "rsync", "-avz",
            "-e", f"ssh -p {self.config.port}" + (f" -i {self.config.key_file}" if self.config.key_file else ""),
            f"{self.config.user}@{self.config.host}:{self.config.workspace}/results/",
            str(local_path) + "/"
        ]

        logger.info(f"下载结果到 {local_dir}...")
        try:
            result = subprocess.run(rsync_cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"❌ 下载失败: {e}")
            return False

    def kill_experiment(self) -> bool:
        """终止实验"""
        code, stdout, _ = self.run_remote(
            "pkill -f 'launch_scientist.py' && echo 'Killed' || echo 'No process'"
        )
        return "Killed" in stdout


class ServerRegistry:
    """服务器注册表"""

    CONFIG_FILE = Path.home() / ".cvpr_auto" / "servers.json"

    def __init__(self):
        self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.servers: Dict[str, ServerConfig] = {}
        self.load()

    def load(self):
        """加载服务器配置"""
        if self.CONFIG_FILE.exists():
            with open(self.CONFIG_FILE) as f:
                data = json.load(f)
                self.servers = {
                    name: ServerConfig.from_dict(cfg)
                    for name, cfg in data.items()
                }

    def save(self):
        """保存服务器配置"""
        data = {
            name: cfg.to_dict()
            for name, cfg in self.servers.items()
        }
        with open(self.CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def add(self, name: str, config: ServerConfig):
        """添加服务器"""
        self.servers[name] = config
        self.save()

    def remove(self, name: str):
        """删除服务器"""
        if name in self.servers:
            del self.servers[name]
            self.save()

    def get(self, name: str) -> Optional[ServerConfig]:
        """获取服务器配置"""
        return self.servers.get(name)

    def list_servers(self) -> List[str]:
        """列出所有服务器"""
        return list(self.servers.keys())


def deploy_to_server(
    server_name: str,
    template: str = "cvpr_lite",
    model: str = "kimi-k2.5",
    **kwargs
) -> bool:
    """一键部署到服务器"""
    registry = ServerRegistry()
    config = registry.get(server_name)

    if not config:
        logger.error(f"❌ 未找到服务器配置: {server_name}")
        return False

    manager = RemoteServerManager(config)

    # 检查连接
    if not manager.check_connection():
        logger.error("❌ 无法连接到服务器，请检查 SSH 配置")
        return False

    # 同步代码
    if not manager.sync_code("."):
        return False

    # 运行实验
    pid = manager.run_experiment(template, model, **kwargs)
    return bool(pid)


if __name__ == "__main__":
    # 测试
    registry = ServerRegistry()

    # 添加测试服务器
    registry.add("local_4090", ServerConfig(
        host="localhost",
        user="test",
        workspace="/tmp/AI-Scientist"
    ))

    print("已配置服务器:", registry.list_servers())
