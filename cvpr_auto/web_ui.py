"""
CVPR-Auto Gradio Web UI
交互式界面用于配置和启动自动科研流程
"""

import gradio as gr
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict
import threading
import time


class ExperimentMonitor:
    """实验进度监控器"""

    def __init__(self):
        self.current_status = "就绪"
        self.log_buffer = []
        self.is_running = False
        self.progress = 0

    def update_status(self, status: str):
        self.current_status = status
        self.log_buffer.append(f"[{time.strftime('%H:%M:%S')}] {status}")

    def get_logs(self) -> str:
        return "\n".join(self.log_buffer[-100:])  # 最近100条


monitor = ExperimentMonitor()


def get_available_templates() -> List[str]:
    """获取可用模板列表"""
    templates_dir = Path("templates")
    if not templates_dir.exists():
        return []
    return [d.name for d in templates_dir.iterdir() if d.is_dir()]


def get_available_models() -> List[str]:
    """获取可用模型列表"""
    return [
        "kimi-k2.5",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
        "deepseek-coder-v2-0724",
        "gemini-1.5-pro",
    ]


def validate_config(
    template: str,
    model: str,
    num_ideas: int,
    max_iterations: int,
    quality_threshold: float,
) -> tuple[bool, str]:
    """验证配置"""
    if not template:
        return False, "请选择一个模板"
    if not model:
        return False, "请选择一个模型"
    if num_ideas < 1 or num_ideas > 50:
        return False, "想法数量必须在 1-50 之间"
    if max_iterations < 1 or max_iterations > 20:
        return False, "迭代次数必须在 1-20 之间"
    if quality_threshold < 1 or quality_threshold > 10:
        return False, "质量阈值必须在 1-10 之间"
    return True, "配置有效"


def start_experiment(
    template: str,
    model: str,
    cvpr_mode: bool,
    paper_driven: bool,
    num_ideas: int,
    max_iterations: int,
    quality_threshold: float,
    keywords: str,
    venues: List[str],
    parallel: int,
    improvement: bool,
    engine: str,
) -> str:
    """启动实验"""
    global monitor

    # 验证配置
    valid, msg = validate_config(template, model, num_ideas, max_iterations, quality_threshold)
    if not valid:
        return f"❌ {msg}"

    if monitor.is_running:
        return "❌ 已有实验正在运行，请等待完成"

    # 构建命令
    cmd = [sys.executable, "launch_scientist.py"]
    cmd.extend(["--experiment", template])
    cmd.extend(["--model", model])
    cmd.extend(["--num-ideas", str(num_ideas)])
    cmd.extend(["--engine", engine])

    if cvpr_mode:
        cmd.append("--cvpr-mode")
        cmd.extend(["--max-iterations", str(max_iterations)])
        cmd.extend(["--quality-threshold", str(quality_threshold)])

    if paper_driven:
        cmd.append("--paper-driven")
        if keywords:
            cmd.extend(["--keywords", keywords])
        if venues:
            cmd.extend(["--venues"] + venues)

    if parallel > 0:
        cmd.extend(["--parallel", str(parallel)])

    if improvement:
        cmd.append("--improvement")

    monitor.is_running = True
    monitor.update_status(f"启动实验: {' '.join(cmd)}")

    # 在后台运行
    def run_experiment():
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in process.stdout:
                monitor.update_status(line.strip())

            process.wait()
            if process.returncode == 0:
                monitor.update_status("✅ 实验完成!")
            else:
                monitor.update_status(f"❌ 实验失败 (返回码: {process.returncode})")

        except Exception as e:
            monitor.update_status(f"❌ 错误: {e}")
        finally:
            monitor.is_running = False

    thread = threading.Thread(target=run_experiment)
    thread.start()

    return f"✅ 实验已启动!\n命令: {' '.join(cmd)}\n\n请切换到'日志监控'标签查看进度"


def get_logs() -> str:
    """获取日志"""
    return monitor.get_logs()


def check_status() -> str:
    """检查状态"""
    status = "🟢 运行中" if monitor.is_running else "⚪ 就绪"
    return f"状态: {status}\n当前: {monitor.current_status}"


def load_results() -> str:
    """加载结果"""
    results_dir = Path("results")
    if not results_dir.exists():
        return "暂无结果"

    output = []
    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir():
            output.append(f"📁 {exp_dir.name}")
            for result_dir in sorted(exp_dir.iterdir()):
                if result_dir.is_dir():
                    # 检查是否有 PDF
                    pdfs = list(result_dir.glob("*.pdf"))
                    pdf_status = "✅" if pdfs else "⏳"
                    output.append(f"  {pdf_status} {result_dir.name}")

    return "\n".join(output) if output else "暂无结果"


def create_ui() -> gr.Blocks:
    """创建 Gradio 界面"""

    with gr.Blocks(title="CVPR-Auto 自动科研系统", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🧠 CVPR-Auto 自动科研系统

        使用 LLM 自动生成研究想法、执行实验、撰写论文
        """)

        with gr.Tab("⚙️ 实验配置"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 基础配置")
                    template = gr.Dropdown(
                        choices=get_available_templates(),
                        label="选择模板",
                        value="cvpr_lite",
                    )
                    model = gr.Dropdown(
                        choices=get_available_models(),
                        label="选择模型",
                        value="kimi-k2.5",
                    )
                    engine = gr.Radio(
                        choices=["openalex", "semanticscholar"],
                        label="文献搜索引擎",
                        value="openalex",
                    )
                    num_ideas = gr.Slider(
                        minimum=1, maximum=20, value=3, step=1,
                        label="生成想法数量"
                    )
                    parallel = gr.Slider(
                        minimum=0, maximum=4, value=0, step=1,
                        label="并行进程数 (0=顺序执行)"
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### CVPR 模式配置")
                    cvpr_mode = gr.Checkbox(
                        label="启用 CVPR 迭代改进模式",
                        value=True,
                    )
                    max_iterations = gr.Slider(
                        minimum=1, maximum=20, value=5, step=1,
                        label="最大迭代次数"
                    )
                    quality_threshold = gr.Slider(
                        minimum=5.0, maximum=9.0, value=7.5, step=0.1,
                        label="质量阈值"
                    )
                    improvement = gr.Checkbox(
                        label="基于评审改进论文",
                        value=True,
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 文献驱动配置")
                    paper_driven = gr.Checkbox(
                        label="使用文献驱动生成想法",
                        value=False,
                    )
                    keywords = gr.Textbox(
                        label="关键词 (逗号分隔)",
                        value="efficient architecture,vision transformer",
                        placeholder="例如: efficient architecture, vision transformer",
                    )
                    venues = gr.CheckboxGroup(
                        choices=["CVPR", "ICCV", "ECCV", "NeurIPS", "ICLR"],
                        label="关注会议",
                        value=["CVPR", "ICCV"],
                    )

            with gr.Row():
                start_btn = gr.Button("🚀 启动实验", variant="primary", size="lg")
                status_text = gr.Textbox(
                    label="启动状态",
                    value="点击上方按钮启动实验",
                    interactive=False,
                )

        with gr.Tab("📊 日志监控"):
            with gr.Row():
                refresh_btn = gr.Button("🔄 刷新日志")
                status_display = gr.Textbox(
                    label="当前状态",
                    value=check_status(),
                    interactive=False,
                )

            logs_display = gr.Textbox(
                label="实验日志",
                value=get_logs,
                lines=30,
                max_lines=50,
                interactive=False,
                every=5,  # 每5秒自动刷新
            )

        with gr.Tab("📁 结果查看"):
            with gr.Row():
                refresh_results_btn = gr.Button("🔄 刷新结果")
            results_display = gr.Textbox(
                label="实验结果",
                value=load_results,
                lines=30,
                interactive=False,
            )

        with gr.Tab("📖 使用指南"):
            gr.Markdown("""
            ## 快速开始

            ### 1. 基础实验（推荐新手）
            - 选择模板: `cvpr_lite` (CIFAR 图像分类)
            - 选择模型: `kimi-k2.5` 或 `claude-3-5-sonnet-20241022`
            - 禁用"CVPR 迭代改进模式"
            - 点击"启动实验"

            ### 2. CVPR 迭代模式
            启用后会执行多轮循环：
            ```
            实验 → 写论文 → 评审 → 改进 → 重复
            ```
            直到论文质量达到阈值或达到最大迭代次数。

            ### 3. 文献驱动模式
            系统自动抓取最新顶会论文：
            - 分析研究空白
            - 基于真实 gaps 生成想法
            - 提高想法的新颖性

            ### 4. 多 GPU 并行
            如果有多个 GPU，可以设置并行进程数，同时运行多个实验。

            ## 模板说明

            | 模板 | 描述 | 数据集 |
            |------|------|--------|
            | cvpr_lite | 轻量级 CVPR 模板 | CIFAR-10/100 |
            | nanoGPT | 语言模型 | OpenWebText |
            | 2d_diffusion | 2D 扩散模型 | 合成数据 |

            ## 注意事项

            1. **API 密钥**: 确保已设置相应的环境变量 (`KIMI_API_KEY`, `OPENAI_API_KEY` 等)
            2. **GPU**: CV 实验需要 GPU，确保 CUDA 可用
            3. **LaTeX**: 论文生成需要安装 `pdflatex` 和 `chktex`
            4. **时间**: 完整实验可能需要数小时，请耐心等待

            ## 结果位置

            生成的论文保存在:
            ```
            results/<template>/<timestamp>_<idea_name>/<idea_name>.pdf
            ```
            """)

        # 事件绑定
        start_btn.click(
            fn=start_experiment,
            inputs=[
                template, model, cvpr_mode, paper_driven,
                num_ideas, max_iterations, quality_threshold,
                keywords, venues, parallel, improvement, engine,
            ],
            outputs=[status_text],
        )

        refresh_btn.click(
            fn=get_logs,
            outputs=[logs_display],
        )

        refresh_results_btn.click(
            fn=load_results,
            outputs=[results_display],
        )

    return app


def main():
    """主函数"""
    print("=" * 70)
    print("🧠 CVPR-Auto Gradio Web UI")
    print("=" * 70)
    print(f"\n可用模板: {get_available_templates()}")
    print(f"\n访问 http://localhost:7860 打开界面")
    print("=" * 70)

    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
