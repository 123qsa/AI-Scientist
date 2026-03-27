import argparse
import json
import multiprocessing
import openai
import os
import os.path as osp
import shutil
import sys
import time
import torch
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from datetime import datetime

from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
from ai_scientist.llm import create_client, AVAILABLE_LLMS
from ai_scientist.perform_experiments import perform_experiments
from ai_scientist.perform_review import perform_review, load_paper, perform_improvement
from ai_scientist.perform_writeup import perform_writeup, generate_latex

NUM_REFLECTIONS = 3


def import_cvpr_modules():
    """动态导入 CVPR 模块，避免循环依赖"""
    try:
        from cvpr_auto.iteration_controller import IterationController
        from cvpr_auto.config import config as cvpr_config
        return IterationController, cvpr_config
    except ImportError:
        return None, None


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI scientist experiments")
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="Skip idea generation and load existing ideas",
    )
    parser.add_argument(
        "--skip-novelty-check",
        action="store_true",
        help="Skip novelty check and use existing ideas",
    )
    # add type of experiment (nanoGPT, Boston, etc.)
    parser.add_argument(
        "--experiment",
        type=str,
        default="nanoGPT",
        help="Experiment to run AI Scientist on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="kimi-k2.5",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--writeup",
        type=str,
        default="latex",
        choices=["latex"],
        help="What format to use for writeup",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Number of parallel processes to run. 0 for sequential execution.",
    )
    parser.add_argument(
        "--improvement",
        action="store_true",
        help="Improve based on reviews.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, all available GPUs will be used.",
    )
    parser.add_argument(
        "--num-ideas",
        type=int,
        default=50,
        help="Number of ideas to generate",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="openalex",
        choices=["semanticscholar", "openalex"],
        help="Scholar engine to use.",
    )
    # CVPR-Auto 模式参数
    parser.add_argument(
        "--cvpr-mode",
        action="store_true",
        help="启用 CVPR-Auto 迭代改进模式（多轮实验-论文-评审-改进）",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="CVPR 模式下的最大迭代次数 (1-20)",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=7.5,
        help="论文质量阈值（低于此分数会触发改进）",
    )
    parser.add_argument(
        "--paper-driven",
        action="store_true",
        help="使用文献驱动的方式生成想法（读取最新论文）",
    )
    parser.add_argument(
        "--venues",
        nargs="+",
        default=["CVPR", "ICCV", "ECCV"],
        help="文献驱动模式下关注的会议",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default="efficient architecture,vision transformer",
        help="文献驱动模式下的关键词（逗号分隔）",
    )
    return parser.parse_args()


def get_available_gpus(gpu_ids=None):
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    return list(range(torch.cuda.device_count()))


def check_latex_dependencies():
    """
    Check if required LaTeX dependencies are installed on the system.
    Returns True if all dependencies are found, False otherwise.
    """
    import shutil
    import sys

    required_dependencies = ['pdflatex', 'chktex']
    missing_deps = []

    for dep in required_dependencies:
        if shutil.which(dep) is None:
            missing_deps.append(dep)
    
    if missing_deps:
        print("Error: Required LaTeX dependencies not found:", file=sys.stderr)
        return False
    
    return True
    
def worker(
        queue,
        base_dir,
        results_dir,
        model,
        client,
        client_model,
        writeup,
        improvement,
        gpu_id,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Worker {gpu_id} started.")
    while True:
        idea = queue.get()
        if idea is None:
            break
        success = do_idea(
            base_dir,
            results_dir,
            idea,
            model,
            client,
            client_model,
            writeup,
            improvement,
            log_file=True,
        )
        print(f"Completed idea: {idea['Name']}, Success: {success}")
    print(f"Worker {gpu_id} finished.")


def do_idea(
        base_dir,
        results_dir,
        idea,
        model,
        client,
        client_model,
        writeup,
        improvement,
        log_file=False,
):
    ## CREATE PROJECT FOLDER
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_name = f"{timestamp}_{idea['Name']}"
    folder_name = osp.join(results_dir, idea_name)
    assert not osp.exists(folder_name), f"Folder {folder_name} already exists."
    destination_dir = folder_name
    shutil.copytree(base_dir, destination_dir, dirs_exist_ok=True)
    with open(osp.join(base_dir, "run_0", "final_info.json"), "r") as f:
        baseline_results = json.load(f)
    # Check if baseline_results is a dictionary before extracting means
    if isinstance(baseline_results, dict):
        baseline_results = {k: v["means"] for k, v in baseline_results.items()}
    exp_file = osp.join(folder_name, "experiment.py")
    vis_file = osp.join(folder_name, "plot.py")
    notes = osp.join(folder_name, "notes.txt")
    with open(notes, "w") as f:
        f.write(f"# Title: {idea['Title']}\n")
        f.write(f"# Experiment description: {idea['Experiment']}\n")
        f.write(f"## Run 0: Baseline\n")
        f.write(f"Results: {baseline_results}\n")
        f.write(f"Description: Baseline results.\n")
    if log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_path = osp.join(folder_name, "log.txt")
        log = open(log_path, "a")
        sys.stdout = log
        sys.stderr = log
    try:
        print_time()
        print(f"*Starting idea: {idea_name}*")
        ## PERFORM EXPERIMENTS
        fnames = [exp_file, vis_file, notes]
        io = InputOutput(
            yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt"
        )
        if model == "deepseek-coder-v2-0724":
            main_model = Model("deepseek/deepseek-coder")
        elif model == "deepseek-reasoner":
            main_model = Model("deepseek/deepseek-reasoner")
        elif model == "llama3.1-405b":
            main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
        elif model.startswith("kimi-"):
            main_model = Model(f"moonshot/{model}")
        else:
            main_model = Model(model)
        coder = Coder.create(
            main_model=main_model,
            fnames=fnames,
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )

        print_time()
        print(f"*Starting Experiments*")
        try:
            success = perform_experiments(idea, folder_name, coder, baseline_results)
        except Exception as e:
            print(f"Error during experiments: {e}")
            print(f"Experiments failed for idea {idea_name}")
            return False

        if not success:
            print(f"Experiments failed for idea {idea_name}")
            return False

        print_time()
        print(f"*Starting Writeup*")
        ## PERFORM WRITEUP
        if writeup == "latex":
            writeup_file = osp.join(folder_name, "latex", "template.tex")
            fnames = [exp_file, writeup_file, notes]
            if model == "deepseek-coder-v2-0724":
                main_model = Model("deepseek/deepseek-coder")
            elif model == "deepseek-reasoner":
                main_model = Model("deepseek/deepseek-reasoner")
            elif model == "llama3.1-405b":
                main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
            elif model.startswith("kimi-"):
                main_model = Model(f"moonshot/{model}")
            else:
                main_model = Model(model)
            coder = Coder.create(
                main_model=main_model,
                fnames=fnames,
                io=io,
                stream=False,
                use_git=False,
                edit_format="diff",
            )
            try:
                perform_writeup(idea, folder_name, coder, client, client_model, engine=args.engine)
            except Exception as e:
                print(f"Failed to perform writeup: {e}")
                return False
            print("Done writeup")
        else:
            raise ValueError(f"Writeup format {writeup} not supported.")

        print_time()
        print(f"*Starting Review*")
        ## REVIEW PAPER
        if writeup == "latex":
            try:
                paper_text = load_paper(f"{folder_name}/{idea['Name']}.pdf")
                review = perform_review(
                    paper_text,
                    model="gpt-4o-2024-05-13",
                    client=openai.OpenAI(),
                    num_reflections=5,
                    num_fs_examples=1,
                    num_reviews_ensemble=5,
                    temperature=0.1,
                )
                # Store the review in separate review.txt file
                with open(osp.join(folder_name, "review.txt"), "w") as f:
                    f.write(json.dumps(review, indent=4))
            except Exception as e:
                print(f"Failed to perform review: {e}")
                return False

        ## IMPROVE WRITEUP
        if writeup == "latex" and improvement:
            print_time()
            print(f"*Starting Improvement*")
            try:
                perform_improvement(review, coder)
                generate_latex(
                    coder, folder_name, f"{folder_name}/{idea['Name']}_improved.pdf"
                )
                paper_text = load_paper(f"{folder_name}/{idea['Name']}_improved.pdf")
                review = perform_review(
                    paper_text,
                    model="gpt-4o-2024-05-13",
                    client=openai.OpenAI(),
                    num_reflections=5,
                    num_fs_examples=1,
                    num_reviews_ensemble=5,
                    temperature=0.1,
                )
                # Store the review in separate review.txt file
                with open(osp.join(folder_name, "review_improved.txt"), "w") as f:
                    f.write(json.dumps(review))
            except Exception as e:
                print(f"Failed to perform improvement: {e}")
                return False
        return True
    except Exception as e:
        print(f"Failed to evaluate idea {idea_name}: {str(e)}")
        return False
    finally:
        print("FINISHED IDEA")
        if log_file:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log.close()


if __name__ == "__main__":
    args = parse_arguments()

    # CVPR 模式：使用文献驱动生成想法 + 迭代改进
    if args.cvpr_mode:
        print("=" * 70)
        print("🎯 CVPR-Auto 模式启动")
        print("=" * 70)

        IterationController, cvpr_config = import_cvpr_modules()
        if IterationController is None:
            print("❌ 无法导入 CVPR 模块，请确保 cvpr_auto/ 目录存在")
            sys.exit(1)

        # 文献驱动生成想法
        if args.paper_driven:
            print("\n📚 使用文献驱动模式生成想法...")
            try:
                from cvpr_auto.paper_tracker import PaperDrivenIdeaPipeline

                pipeline = PaperDrivenIdeaPipeline()
                keywords = [k.strip() for k in args.keywords.split(",")]
                gaps, ideas = pipeline.run_full_pipeline(
                    venues=args.venues,
                    keywords=keywords,
                    days=90,
                    max_papers=50
                )

                # 转换想法格式
                novel_ideas = []
                for i, idea in enumerate(ideas[:args.num_ideas]):
                    novel_ideas.append({
                        "Name": f"cvpr_idea_{i+1}",
                        "Title": idea.get("title", "Untitled"),
                        "Experiment": idea.get("solution", ""),
                        "novel": True,
                        "idea": idea
                    })

                print(f"\n✓ 文献驱动生成了 {len(novel_ideas)} 个想法")

            except Exception as e:
                print(f"⚠️ 文献驱动失败，回退到标准模式: {e}")
                args.paper_driven = False

        # 标准想法生成（如果不是文献驱动或文献驱动失败）
        if not args.paper_driven:
            client, client_model = create_client(args.model)
            base_dir = osp.join("templates", args.experiment)
            results_dir = osp.join("results", args.experiment)

            ideas = generate_ideas(
                base_dir,
                client=client,
                model=client_model,
                skip_generation=args.skip_idea_generation,
                max_num_generations=args.num_ideas,
                num_reflections=NUM_REFLECTIONS,
            )
            if not args.skip_novelty_check:
                ideas = check_idea_novelty(
                    ideas,
                    base_dir=base_dir,
                    client=client,
                    model=client_model,
                    engine=args.engine,
                )

            with open(osp.join(base_dir, "ideas.json"), "w") as f:
                json.dump(ideas, f, indent=4)

            novel_ideas = [idea for idea in ideas if idea.get("novel", True)]

        # 启动 CVPR 迭代流程
        print(f"\n🚀 启动 CVPR 迭代改进流程（最多 {args.max_iterations} 轮）")

        for idea in novel_ideas:
            print(f"\n{'='*70}")
            print(f"处理想法: {idea['Name']}")
            print(f"{'='*70}")

            try:
                # 创建输出目录
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                idea_folder = f"{timestamp}_{idea['Name']}"
                output_dir = osp.join("results", "cvpr", idea_folder)
                os.makedirs(output_dir, exist_ok=True)

                # 初始化迭代控制器
                controller = IterationController(
                    base_dir=osp.join("templates", args.experiment),
                    output_dir=output_dir,
                    idea=idea,
                    llm_client=client if 'client' in locals() else create_client(args.model)[0],
                    quality_thresholds={
                        "novelty_score": args.quality_threshold,
                        "experiment_rigor": args.quality_threshold,
                        "writing_quality": args.quality_threshold,
                        "min_improvement": 1.0,
                    },
                    max_iterations=args.max_iterations,
                    remote_server=None  # 本地运行，如需远程可配置
                )

                # 运行迭代循环
                final_results = controller.run_iteration_loop()

                print(f"\n✓ 想法 {idea['Name']} 完成")
                print(f"  最终质量分数: {final_results.get('final_quality', 'N/A')}")
                print(f"  总迭代次数: {final_results.get('iterations', 0)}")
                print(f"  输出目录: {output_dir}")

            except Exception as e:
                print(f"❌ 处理想法 {idea['Name']} 失败: {e}")
                import traceback
                print(traceback.format_exc())
                continue

        print("\n" + "=" * 70)
        print("CVPR-Auto 模式完成！")
        print("=" * 70)
        sys.exit(0)

    # 标准模式（原有逻辑）
    # Check available GPUs and adjust parallel processes if necessary
    available_gpus = get_available_gpus(args.gpus)
    if args.parallel > len(available_gpus):
        print(
            f"Warning: Requested {args.parallel} parallel processes, but only {len(available_gpus)} GPUs available. Adjusting to {len(available_gpus)}."
        )
        args.parallel = len(available_gpus)

    print(f"Using GPUs: {available_gpus}")

    # Check LaTeX dependencies before proceeding
    if args.writeup == "latex" and not check_latex_dependencies():
        sys.exit(1)

    # Create client
    client, client_model = create_client(args.model)

    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)
    ideas = generate_ideas(
        base_dir,
        client=client,
        model=client_model,
        skip_generation=args.skip_idea_generation,
        max_num_generations=args.num_ideas,
        num_reflections=NUM_REFLECTIONS,
    )
    if not args.skip_novelty_check:
        ideas = check_idea_novelty(
            ideas,
            base_dir=base_dir,
            client=client,
            model=client_model,
            engine=args.engine,
        )

    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)

    novel_ideas = [idea for idea in ideas if idea["novel"]]
    # novel_ideas = list(reversed(novel_ideas))

    if args.parallel > 0:
        print(f"Running {args.parallel} parallel processes")
        queue = multiprocessing.Queue()
        for idea in novel_ideas:
            queue.put(idea)

        processes = []
        for i in range(args.parallel):
            gpu_id = available_gpus[i % len(available_gpus)]
            p = multiprocessing.Process(
                target=worker,
                args=(
                    queue,
                    base_dir,
                    results_dir,
                    args.model,
                    client,
                    client_model,
                    args.writeup,
                    args.improvement,
                    gpu_id,
                ),
            )
            p.start()
            time.sleep(150)
            processes.append(p)

        # Signal workers to exit
        for _ in range(args.parallel):
            queue.put(None)

        for p in processes:
            p.join()

        print("All parallel processes completed.")
    else:
        for idea in novel_ideas:
            print(f"Processing idea: {idea['Name']}")
            try:
                success = do_idea(
                    base_dir,
                    results_dir,
                    idea,
                    args.model,
                    client,
                    client_model,
                    args.writeup,
                    args.improvement,
                )
                print(f"Completed idea: {idea['Name']}, Success: {success}")
            except Exception as e:
                print(f"Failed to evaluate idea {idea['Name']}: {str(e)}")
                import traceback
                print(traceback.format_exc())
    print("All ideas evaluated.")
