"""
AI-Scientist Web UI
基于 Gradio 的图形化界面
"""

import gradio as gr
import subprocess
import os
import json
import sys
from pathlib import Path

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

# 运行 AI-Scientist
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
    """运行 AI-Scientist 实验"""
    cmd = [sys.executable, "launch_scientist.py"]

    # 构建命令
    cmd.extend(["--model", model])
    cmd.extend(["--experiment", experiment])
    cmd.extend(["--num-ideas", str(num_ideas)])
    cmd.extend(["--engine", engine])

    if parallel > 0:
        cmd.extend(["--parallel", str(parallel)])
    if improvement:
        cmd.append("--improvement")
    if skip_idea_generation:
        cmd.append("--skip-idea-generation")
    if skip_novelty_check:
        cmd.append("--skip-novelty-check")
    if gpus:
        cmd.extend(["--gpus", gpus])

    # 运行命令并捕获输出
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
            yield "".join(output) + "\n\n✅ 实验完成！"
        else:
            yield "".join(output) + "\n\n❌ 实验失败！"

    except Exception as e:
        yield f"错误: {str(e)}"

# 检查 baseline
def check_baseline(experiment):
    """检查模板是否已准备 baseline"""
    baseline_path = Path(f"templates/{experiment}/run_0/final_info.json")
    if baseline_path.exists():
        return f"✅ {experiment} 已准备 baseline"
    else:
        return f"⚠️ {experiment} 缺少 baseline，请先运行: cd templates/{experiment} && python experiment.py --out_dir run_0"

# 准备 baseline
def prepare_baseline(experiment):
    """为模板准备 baseline"""
    import threading

    def run_baseline():
        template_dir = Path(f"templates/{experiment}")
        if not template_dir.exists():
            return f"❌ 模板 {experiment} 不存在"

        try:
            # 运行实验
            result1 = subprocess.run(
                [sys.executable, "experiment.py", "--out_dir", "run_0"],
                cwd=str(template_dir),
                capture_output=True,
                text=True
            )

            if result1.returncode != 0:
                return f"❌ experiment.py 失败:\n{result1.stderr}"

            # 运行绘图
            result2 = subprocess.run(
                [sys.executable, "plot.py"],
                cwd=str(template_dir),
                capture_output=True,
                text=True
            )

            return f"✅ {experiment} baseline 准备完成！\n\n{result1.stdout}\n{result2.stdout}"

        except Exception as e:
            return f"❌ 错误: {str(e)}"

    return run_baseline()

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

    with gr.Blocks(title="AI-Scientist 控制台", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 AI-Scientist 控制台

        全自动科学发现系统 - 基于大语言模型生成研究想法、运行实验、撰写论文
        """)

        with gr.Tab("🚀 运行实验"):
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
            ## AI-Scientist Web UI

            ### 快速开始
            1. 选择模型和实验模板
            2. 确保模板已准备 baseline（点击"检查 baseline"）
            3. 如未准备，点击"准备 baseline"
            4. 配置参数后点击"启动 AI-Scientist"

            ### 模板说明
            - **nanoGPT_lite**: 轻量级语言模型实验
            - **2d_diffusion**: 2D 扩散模型
            - **grokking**: 模型泛化能力研究

            ### 注意事项
            - 首次运行需要下载模型和数据，耗时较长
            - 生成的论文保存在 `results/<模板>/<时间戳>_<想法>/report.pdf`
            - 运行过程中请勿关闭浏览器

            ### 文档
            - 项目文档: [CLAUDE.md](CLAUDE.md)
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
        show_error=True
    )
