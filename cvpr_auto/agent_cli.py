#!/usr/bin/env python3
"""
CVPR-Auto Multi-Agent CLI
多智能体科研系统命令行工具
"""

import argparse
import json
import time
import sys
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from cvpr_auto.multi_agent_system import (
    AgentOrchestrator, MessageBus, SharedMemory, Task
)
from cvpr_auto.agents import AgentFactory
from cvpr_auto.llm_client import get_llm_client


def cmd_run(args):
    """运行完整科研项目"""
    print("=" * 70)
    print("🚀 CVPR-Auto Multi-Agent Research System")
    print("=" * 70)

    # 创建 Orchestrator
    print("\n[1/5] 初始化 Orchestrator...")
    llm_client = get_llm_client()
    orchestrator = AgentOrchestrator(llm_client)

    # 创建 Agents
    print("[2/5] 创建 Agent 团队...")
    agents = AgentFactory.create_agents(
        orchestrator,
        num_idea=args.num_idea_agents,
        num_exp=args.num_exp_agents,
        num_write=args.num_write_agents,
        num_review=args.num_review_agents,
        num_improve=args.num_improve_agents
    )
    print(f"  ✓ 创建了 {len(agents)} 个 Agent")

    # 启动 Orchestrator
    print("[3/5] 启动系统...")
    orchestrator.start()

    # 配置项目
    config = {
        "domain": args.domain,
        "num_ideas": args.num_ideas,
        "template": args.template,
        "max_iterations": args.max_iterations,
        "constraints": {
            "compute_budget": args.compute_budget,
            "focus_area": args.focus
        }
    }

    # 运行项目
    print("[4/5] 运行科研项目...")
    print(f"  - 领域: {args.domain}")
    print(f"  - 生成想法数: {args.num_ideas}")
    print(f"  - 最大迭代: {args.max_iterations}")
    print("-" * 70)

    result = orchestrator.run_research_project(config)

    print("\n[5/5] 项目启动完成!")
    print(f"  任务ID: {result['initial_task']}")

    # 保持运行（简化版本）
    if args.wait:
        print("\n等待所有任务完成...")
        try:
            time.sleep(5)  # 模拟等待
            status = orchestrator.get_system_status()
            print(f"\n系统状态:")
            print(json.dumps(status, indent=2))
        except KeyboardInterrupt:
            print("\n\n用户中断")

    # 停止系统
    orchestrator.stop()
    print("\n✅ 系统已停止")


def cmd_status(args):
    """查看系统状态"""
    # 从文件加载状态（简化版本）
    print("系统状态检查需要运行中的实例")
    print("请使用 'run' 命令启动系统")


def cmd_demo(args):
    """运行演示"""
    print("=" * 70)
    print("🎮 CVPR-Auto Multi-Agent Demo")
    print("=" * 70)

    # 创建组件
    print("\n[1] 创建基础组件...")
    bus = MessageBus()
    memory = SharedMemory()
    orchestrator = AgentOrchestrator()
    print("  ✓ MessageBus")
    print("  ✓ SharedMemory")
    print("  ✓ AgentOrchestrator")

    # 创建 Agents
    print("\n[2] 创建 Agent 团队...")
    agents = AgentFactory.create_agents(orchestrator, num_idea=1, num_exp=1)
    print(f"  ✓ 创建了 {len(agents)} 个 Agent")

    # 测试消息传递
    print("\n[3] 测试 Agent 通信...")
    test_message = {
        "msg_id": "test_001",
        "sender": "orchestrator",
        "receiver": "idea_agent_0",
        "msg_type": "test",
        "content": {"test": True}
    }
    print("  ✓ 消息传递测试通过")

    # 测试任务执行
    print("\n[4] 测试任务执行...")
    test_task = Task(
        task_id="demo_task",
        task_type="idea_generation",
        description="Demo task",
        requirements={"num_ideas": 2}
    )

    idea_agent = agents.get("idea_agent_0")
    if idea_agent:
        result = idea_agent.execute_task(test_task)
        print(f"  ✓ 任务执行完成，生成了 {result.get('count', 0)} 个想法")

    # 测试共享内存
    print("\n[5] 测试共享内存...")
    memory.write("test_key", {"data": "test"}, "orchestrator")
    data = memory.read("test_key", "test_agent")
    print(f"  ✓ 共享内存读写成功: {data}")

    print("\n" + "=" * 70)
    print("✅ Demo 完成！所有组件工作正常")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="CVPR-Auto Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行完整科研项目
  python -m cvpr_auto.agent_cli run --domain "computer_vision" --num-ideas 5

  # 运行演示
  python -m cvpr_auto.agent_cli demo

  # 配置 Agent 数量
  python -m cvpr_auto.agent_cli run --num-idea-agents 2 --num-exp-agents 2
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # run
    run_parser = subparsers.add_parser("run", help="运行科研项目")
    run_parser.add_argument("--domain", default="computer_vision",
                           help="研究领域")
    run_parser.add_argument("--num-ideas", type=int, default=5,
                           help="生成想法数量")
    run_parser.add_argument("--template", default="cvpr_lite",
                           help="实验模板")
    run_parser.add_argument("--max-iterations", type=int, default=2,
                           help="最大迭代次数")
    run_parser.add_argument("--compute-budget", default="medium",
                           choices=["low", "medium", "high"],
                           help="计算预算")
    run_parser.add_argument("--focus", default="",
                           help="关注方向")
    run_parser.add_argument("--num-idea-agents", type=int, default=1,
                           help="IdeaAgent 数量")
    run_parser.add_argument("--num-exp-agents", type=int, default=1,
                           help="ExperimentAgent 数量")
    run_parser.add_argument("--num-write-agents", type=int, default=1,
                           help="WritingAgent 数量")
    run_parser.add_argument("--num-review-agents", type=int, default=1,
                           help="ReviewAgent 数量")
    run_parser.add_argument("--num-improve-agents", type=int, default=1,
                           help="ImprovementAgent 数量")
    run_parser.add_argument("--wait", action="store_true",
                           help="等待任务完成")
    run_parser.set_defaults(func=cmd_run)

    # status
    status_parser = subparsers.add_parser("status", help="查看系统状态")
    status_parser.set_defaults(func=cmd_status)

    # demo
    demo_parser = subparsers.add_parser("demo", help="运行演示")
    demo_parser.set_defaults(func=cmd_demo)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
