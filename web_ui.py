"""
AI-Scientist Web UI
基于 Gradio 的图形化界面 - 默认云端执行模式
"""

import gradio as gr
import subprocess
import os
import json
import sys
from pathlib import Path

# 导入远程执行模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remote_runner import (
    run_experiment_on_server,
    prepare_baseline_on_server,
    check_baseline_on_server,
    check_server_connection,
    sync_results
)

# 获取可用模板
def get_templates():
    templates_dir = Path("templates")
    if not templates_dir.exists():
        return []
    return [d.name for d in templates_dir.iterdir() if d.is_dir()]

# 获取可用模型
def get_models():
    from ai_scientist.llm import AVAILABLE_LLMS
    return AVAILABLE_LLMS

# 获取服务器状态
def get_server_status():
    """获取服务器连接状态"""
    if check_server_connection():
        return "🟢 云端服务器已连接"
    else:
        return "🔴 云端服务器未连接"

# 运行 AI-Scientist（云端）
def run_ai_scientist(
    model,
    experiment,
    num_ideas,
    engine,
    parallel,
    improvement,
    skip_idea_generation,
    skip_novelty_check,
    gpus
):
    """在云端服务器运行 AI-Scientist 实验"""

    # 检查服务器
    if not check_server_connection():
        yield "❌ 无法连接云端服务器\n"
        yield "请检查:\n"
        yield "  1. VPN/网络连接\n"
        yield "  2. SSH 密钥配置\n"
        yield f"  3. 服务器地址: 166.111.86.21\n"
        return

    yield f"☁️  正在连接云端服务器...\n"
    yield f"🚀 启动远程实验...\n"
    yield f"   模型: {model}\n"
    yield f"   模板: {experiment}\n"
    yield f"   想法数: {num_ideas}\n"
    yield "-" * 60 + "\n"

    # 构建参数
    kwargs = {
        "skip_idea_generation": skip_idea_generation,
        "skip_novelty_check": skip_novelty_check,
        "improvement": improvement,
    }
    if gpus:
        kwargs["gpus"] = gpus

    # 运行实验
    import threading
    import queue

    output_queue = queue.Queue()

    def run_remote():
        try:
            result = run_experiment_on_server(
                model=model,
                experiment=experiment,
                num_ideas=num_ideas,
                engine=engine,
                parallel=parallel,
                **kwargs
            )
            output_queue.put(("done", result))
        except Exception as e:
            output_queue.put(("error", str(e)))

    # 启动远程线程
    thread = threading.Thread(target=run_remote)
    thread.daemon = True
    thread.start()

    # 等待完成
    thread.join()

    if not output_queue.empty():
        status, result = output_queue.get()
        if status == "done":
            yield f"\n{'='*60}\n✅ 实验完成！结果已同步到本地 results/ 目录"
        else:
            yield f"\n❌ 错误: {result}"

# 检查 baseline（云端）
def check_baseline(experiment):
    """检查云端服务器上的 baseline"""
    if not check_server_connection():
        return "🔴 服务器未连接"
    return check_baseline_on_server(experiment)

# 准备 baseline（云端）
def prepare_baseline(experiment):
    """在云端服务器准备 baseline"""
    if not check_server_connection():
        return "❌ 无法连接服务器"

    yield f"📡 在云端准备 {experiment} baseline...\n"

    import subprocess
    from remote_runner import get_ssh_base_cmd, REMOTE_PROJECT_DIR

    cmd = get_ssh_base_cmd() + [
        f"cd {REMOTE_PROJECT_DIR}/templates/{experiment} && "
        f"python experiment.py --out_dir run_0 2>&1"
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        output = []
        for line in process.stdout:
            output.append(line)
            yield "".join(output)

        process.wait()

        if process.returncode == 0:
            yield "".join(output) + "\n\n✅ baseline 准备完成！"
        else:
            yield "".join(output) + "\n\n❌ baseline 准备失败！"

    except Exception as e:
        yield f"错误: {str(e)}"

# 查看结果
def list_results():
    """列出所有生成的论文"""
    results_dir = Path("results")
    if not results_dir.exists():
        return "暂无结果"

    results = []
    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir():
            results.append(f"\n📁 {exp_dir.name}")
            for run_dir in sorted(exp_dir.iterdir()):
                if run_dir.is_dir():
                    pdf_path = run_dir / "report.pdf"
                    if pdf_path.exists():
                        results.append(f"  ✅ {run_dir.name}/report.pdf")
                    else:
                        results.append(f"  ⏳ {run_dir.name}/ (进行中)")

    return "\n".join(results) if results else "暂无结果"

# 创建界面
def create_ui():
    templates = get_templates()
    models = get_models()

    with gr.Blocks(title="AI-Scientist 控制台") as demo:
        gr.Markdown("""
        # 🤖 AI-Scientist 控制台

        全自动科学发现系统 - 基于大语言模型生成研究想法、运行实验、撰写论文
        """)

        with gr.Tab("🚀 运行实验（云端）"):
            # 服务器状态
            gr.Markdown(f"""
            ### ☁️ 云端执行模式
            所有实验将在远程服务器运行 (166.111.86.21)
            **状态**: {get_server_status()}
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 配置")

                    model_dropdown = gr.Dropdown(
                        choices=models,
                        value="kimi-k2.5",
                        label="模型",
                        info="选择用于生成想法和代码的 LLM"
                    )

                    experiment_dropdown = gr.Dropdown(
                        choices=templates,
                        value=templates[0] if templates else None,
                        label="实验模板",
                        info="选择研究领域模板"
                    )

                    check_btn = gr.Button("检查 baseline", variant="secondary")
                    baseline_status = gr.Textbox(
                        label="状态",
                        value=lambda: check_baseline(templates[0]) if templates else "无模板",
                        interactive=False
                    )

                    prepare_btn = gr.Button("准备 baseline", variant="secondary")
                    prepare_output = gr.Textbox(label="准备结果", lines=5)

                with gr.Column(scale=1):
                    num_ideas = gr.Slider(
                        minimum=1, maximum=20, value=2, step=1,
                        label="生成想法数量",
                        info="每个想法会生成一篇独立论文"
                    )

                    engine = gr.Radio(
                        choices=["openalex", "semanticscholar"],
                        value="openalex",
                        label="文献检索引擎"
                    )

                    parallel = gr.Slider(
                        minimum=0, maximum=8, value=0, step=1,
                        label="并行进程数",
                        info="0 = 顺序执行，>0 = 并行（需要多 GPU）"
                    )

                    with gr.Accordion("高级选项", open=False):
                        improvement = gr.Checkbox(
                            label="基于评审改进论文",
                            value=False
                        )
                        skip_idea = gr.Checkbox(
                            label="跳过想法生成（使用已有）",
                            value=False
                        )
                        skip_novelty = gr.Checkbox(
                            label="跳过新颖性检查",
                            value=False
                        )
                        gpus = gr.Textbox(
                            label="指定 GPU",
                            placeholder="例如: 0,1,2（留空使用所有 GPU）"
                        )

                with gr.Column(scale=2):
                    gr.Markdown("### 输出")
                    run_btn = gr.Button("🚀 启动 AI-Scientist", variant="primary", size="lg")
                    output = gr.Textbox(
                        label="运行日志",
                        lines=25,
                        max_lines=50,
                        autoscroll=True
                    )

        with gr.Tab("📊 结果"):
            refresh_btn = gr.Button("刷新结果列表")
            results_list = gr.Textbox(
                label="生成的论文",
                lines=20,
                value=list_results
            )

        with gr.Tab("ℹ️ 关于"):
            gr.Markdown("""
            ## AI-Scientist Web UI - 云端版

            ### 执行模式
            **☁️ 云端模式（默认）**: 所有实验通过 SSH 在远程服务器运行
            - 服务器: `hanjiajun@166.111.86.21`
            - 本地仅作为控制台显示输出
            - 结果自动同步到本地 `results/` 目录

            ### 快速开始
            1. 确保 VPN 连接正常（服务器在校园网）
            2. 选择模型和实验模板
            3. 点击"检查 baseline"确保服务器已准备
            4. 配置参数后点击"启动 AI-Scientist"
            5. 等待实验完成，结果自动同步

            ### 本地调试模式
            ```bash
            LOCAL_MODE=1 python launch_scientist_remote.py
            ```

            ### 模板说明
            - **nanoGPT_lite**: 轻量级语言模型实验
            - **2d_diffusion**: 2D 扩散模型
            - **grokking**: 模型泛化能力研究

            ### 文档
            - 项目文档: [CLAUDE.md](CLAUDE.md)
            - 服务器配置: [SERVER_CONFIG.md](SERVER_CONFIG.md)
            - 原论文: https://arxiv.org/abs/2408.06292
            """)

        # 事件绑定
        check_btn.click(
            fn=check_baseline,
            inputs=[experiment_dropdown],
            outputs=[baseline_status]
        )

        experiment_dropdown.change(
            fn=check_baseline,
            inputs=[experiment_dropdown],
            outputs=[baseline_status]
        )

        prepare_btn.click(
            fn=prepare_baseline,
            inputs=[experiment_dropdown],
            outputs=[prepare_output]
        )

        run_btn.click(
            fn=run_ai_scientist,
            inputs=[
                model_dropdown,
                experiment_dropdown,
                num_ideas,
                engine,
                parallel,
                improvement,
                skip_idea,
                skip_novelty,
                gpus
            ],
            outputs=[output]
        )

        refresh_btn.click(
            fn=list_results,
            outputs=[results_list]
        )

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft()
    )
