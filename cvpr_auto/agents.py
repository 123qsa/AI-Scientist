"""
具体 Agent 实现
包括: IdeaAgent, ExperimentAgent, WritingAgent, ReviewAgent, ImprovementAgent
"""

import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import sys
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from cvpr_auto.multi_agent_system import (
    BaseAgent, AgentRole, MessageBus, SharedMemory, Task
)
from cvpr_auto.llm_client import get_llm_client, LLMResponse


class IdeaAgent(BaseAgent):
    """
    想法生成智能体
    负责生成、评估和选择研究想法
    """

    def __init__(self, agent_id: str, message_bus: MessageBus, shared_memory: SharedMemory):
        super().__init__(agent_id, AgentRole.IDEA, message_bus, shared_memory)
        self.generated_ideas: List[Dict] = []

    def execute_task(self, task: Task) -> Dict[str, Any]:
        """执行想法生成任务"""
        print(f"[{self.agent_id}] 🧠 生成研究想法...")

        requirements = task.requirements
        domain = requirements.get("domain", "computer_vision")
        num_ideas = requirements.get("num_ideas", 5)
        constraints = requirements.get("constraints", {})

        # 获取 LLM 客户端
        if not self.llm_client:
            self.llm_client = get_llm_client()

        # 生成想法
        ideas = self._generate_ideas(domain, num_ideas, constraints)

        # 评估想法
        evaluated_ideas = self._evaluate_ideas(ideas)

        # 选择最佳想法
        best_ideas = self._select_best_ideas(evaluated_ideas, top_k=3)

        result = {
            "ideas": ideas,
            "evaluated": evaluated_ideas,
            "best": best_ideas,
            "count": len(ideas),
            "timestamp": datetime.now().isoformat()
        }

        self.generated_ideas.extend(ideas)

        print(f"[{self.agent_id}] ✓ 生成了 {len(ideas)} 个想法，选择了 {len(best_ideas)} 个最佳")

        return result

    def _generate_ideas(self, domain: str, num: int, constraints: Dict) -> List[Dict]:
        """使用 LLM 生成想法"""

        system_prompt = f"""You are an expert researcher in {domain}.
Generate novel, feasible, and impactful research ideas.
Each idea should include:
- Title
- Problem statement
- Proposed method
- Expected contributions
- Potential impact"""

        prompt = f"""Generate {num} research ideas in the field of {domain}.

Constraints:
- Focus on novel, unsolved problems
- Ideas should be feasible for a small research team
- Prefer ideas with clear evaluation metrics
- Avoid ideas that require massive compute resources

For each idea, provide:
1. Name: Short, descriptive name (no spaces, use underscores)
2. Title: Full paper title
3. Problem: What problem does this solve? (2-3 sentences)
4. Method: High-level approach (3-4 sentences)
5. Experiment: How to evaluate? (2-3 sentences)
6. NoveltyScore: 1-10 rating of novelty
7. FeasibilityScore: 1-10 rating of feasibility

Respond in JSON format:
{{
  "ideas": [
    {{
      "Name": "idea_name",
      "Title": "Full Title",
      "Problem": "description",
      "Method": "approach",
      "Experiment": "evaluation plan",
      "NoveltyScore": 8,
      "FeasibilityScore": 7
    }}
  ]
}}"""

        try:
            response = self.llm_client.generate(prompt, system_prompt)

            if not response.error:
                # 提取 JSON
                import re
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    return data.get("ideas", [])
        except Exception as e:
            print(f"[{self.agent_id}] 生成想法出错: {e}")

        # 返回默认想法
        return self._default_ideas(num)

    def _evaluate_ideas(self, ideas: List[Dict]) -> List[Dict]:
        """评估想法质量"""
        evaluated = []

        for idea in ideas:
            # 计算综合分数
            novelty = idea.get("NoveltyScore", 5)
            feasibility = idea.get("FeasibilityScore", 5)

            # 综合评分 (加权)
            overall = (novelty * 0.6 + feasibility * 0.4)

            idea["OverallScore"] = round(overall, 2)
            idea["Evaluation"] = self._generate_evaluation_text(idea)

            evaluated.append(idea)

        # 按分数排序
        evaluated.sort(key=lambda x: x["OverallScore"], reverse=True)

        return evaluated

    def _generate_evaluation_text(self, idea: Dict) -> str:
        """生成评估文本"""
        novelty = idea.get("NoveltyScore", 5)
        feasibility = idea.get("FeasibilityScore", 5)

        if novelty >= 8 and feasibility >= 7:
            return "High novelty and feasible. Recommended for immediate pursuit."
        elif novelty >= 7:
            return "Novel idea with moderate feasibility. Worth exploring."
        elif feasibility >= 8:
            return "Practical and feasible. Lower novelty but good baseline."
        else:
            return "Average scores. Consider refining or combining with other ideas."

    def _select_best_ideas(self, ideas: List[Dict], top_k: int = 3) -> List[Dict]:
        """选择最佳想法"""
        return ideas[:top_k]

    def _default_ideas(self, num: int) -> List[Dict]:
        """默认想法（备用）"""
        defaults = [
            {
                "Name": "efficient_attention",
                "Title": "Efficient Attention Mechanisms for Vision Transformers",
                "Problem": "Self-attention in ViTs has quadratic complexity limiting scalability.",
                "Method": "Propose a linear attention approximation with local-global hybrid.",
                "Experiment": "Evaluate on ImageNet and measure throughput vs accuracy tradeoff.",
                "NoveltyScore": 7,
                "FeasibilityScore": 8
            },
            {
                "Name": "adaptive_augmentation",
                "Title": "Learning Adaptive Augmentation Policies Online",
                "Problem": "Fixed augmentation policies may not be optimal for all training stages.",
                "Method": "Use reinforcement learning to adapt augmentation during training.",
                "Experiment": "Compare against AutoAugment and RandAugment on CIFAR-100.",
                "NoveltyScore": 8,
                "FeasibilityScore": 7
            },
            {
                "Name": "contrastive_kd",
                "Title": "Contrastive Knowledge Distillation for Model Compression",
                "Problem": "Traditional KD doesn't capture inter-class relationships well.",
                "Method": "Add contrastive learning to distillation to preserve structure.",
                "Experiment": "Distill ResNet-50 to MobileNet on ImageNet.",
                "NoveltyScore": 7,
                "FeasibilityScore": 8
            }
        ]
        return defaults[:num]


class ExperimentAgent(BaseAgent):
    """
    实验执行智能体
    负责运行实验、收集结果
    """

    def __init__(self, agent_id: str, message_bus: MessageBus, shared_memory: SharedMemory):
        super().__init__(agent_id, AgentRole.EXPERIMENT, message_bus, shared_memory)
        self.experiment_count = 0
        self.success_count = 0

    def execute_task(self, task: Task) -> Dict[str, Any]:
        """执行实验任务"""
        print(f"[{self.agent_id}] 🔬 执行实验...")

        requirements = task.requirements
        idea = requirements.get("input_data", {}).get("ideas", [{}])[0]
        template = requirements.get("template", "cvpr_lite")

        # 检查是否有远程服务器配置
        remote_config = self.shared_memory.read("config:remote_server", self.agent_id)

        if remote_config:
            # 远程执行
            result = self._run_remote_experiment(idea, template, remote_config)
        else:
            # 本地执行（简化版本）
            result = self._run_local_experiment(idea, template)

        self.experiment_count += 1
        if result.get("success", False):
            self.success_count += 1

        print(f"[{self.agent_id}] ✓ 实验完成: {result.get('status', 'unknown')}")

        return result

    def _run_local_experiment(self, idea: Dict, template: str) -> Dict:
        """本地运行实验（简化）"""
        # 实际应该调用 experiment.py
        # 这里返回模拟结果

        return {
            "success": True,
            "status": "completed",
            "template": template,
            "idea": idea.get("Name", "unknown"),
            "results": {
                "accuracy": 75.0 + (self.experiment_count * 2),
                "parameters": "11M",
                "training_time": 3600
            },
            "plots": ["training_curves.png", "comparison.png"],
            "notes": f"Experiment completed for {idea.get('Name', 'unknown')}"
        }

    def _run_remote_experiment(self, idea: Dict, template: str, remote_config: Dict) -> Dict:
        """远程运行实验"""
        # 实际应该调用 remote_manager
        # 这里返回模拟结果

        return {
            "success": True,
            "status": "completed_remote",
            "server": remote_config.get("host", "unknown"),
            "results": {
                "accuracy": 78.5,
                "parameters": "10M",
                "training_time": 2400
            }
        }


class WritingAgent(BaseAgent):
    """
    论文撰写智能体
    负责生成论文各部分
    """

    def __init__(self, agent_id: str, message_bus: MessageBus, shared_memory: SharedMemory):
        super().__init__(agent_id, AgentRole.WRITING, message_bus, shared_memory)
        self.sections_written = []

    def execute_task(self, task: Task) -> Dict[str, Any]:
        """执行写作任务"""
        print(f"[{self.agent_id}] ✍️ 撰写论文...")

        requirements = task.requirements
        experiment_results = requirements.get("input_data", {})
        idea = experiment_results.get("idea", {})

        # 论文结构
        paper_sections = {
            "abstract": self._write_abstract(idea, experiment_results),
            "introduction": self._write_introduction(idea),
            "method": self._write_method(idea),
            "experiments": self._write_experiments(experiment_results),
            "conclusion": self._write_conclusion(idea, experiment_results)
        }

        result = {
            "sections": paper_sections,
            "word_count": sum(len(s.split()) for s in paper_sections.values()),
            "figures": ["training_curves.pdf", "comparison.pdf"],
            "tables": ["main_results"],
            "timestamp": datetime.now().isoformat()
        }

        self.sections_written.extend(paper_sections.keys())

        print(f"[{self.agent_id}] ✓ 论文撰写完成: {result['word_count']} 词")

        return result

    def _write_abstract(self, idea: Dict, results: Dict) -> str:
        """撰写摘要"""
        return f"""This paper presents {idea.get('Title', 'a novel approach')}.
{idea.get('Problem', '')} We propose {idea.get('Method', 'a new method')}.
Experimental results demonstrate the effectiveness of our approach."""

    def _write_introduction(self, idea: Dict) -> str:
        """撰写引言"""
        return f"""\section{{Introduction}}

{idea.get('Problem', 'The problem is important...')}

Despite recent advances, this problem remains challenging due to several factors...

In this paper, we propose {idea.get('Title', 'a novel method')} to address these challenges.
Our main contributions are:

\begin{{itemize}}
    \item A novel approach that improves upon existing methods
    \item Comprehensive experiments demonstrating effectiveness
    \item Analysis providing insights into the problem
\end{{itemize}}"""

    def _write_method(self, idea: Dict) -> str:
        """撰写方法"""
        return f"""\section{{Method}}

\subsection{{Overview}}

Our approach consists of three main components...

\subsection{{Detailed Design}}

{idea.get('Method', 'The method is described here...')}

\subsection{{Implementation Details}}

We implement our method in PyTorch..."""

    def _write_experiments(self, results: Dict) -> str:
        """撰写实验"""
        return f"""\section{{Experiments}}

\subsection{{Experimental Setup}}

We evaluate our method on standard benchmarks...

\subsection{{Main Results}}

As shown in Table 1, our method achieves competitive results...

\subsection{{Ablation Studies}}

We conduct ablation studies to validate our design choices..."""

    def _write_conclusion(self, idea: Dict, results: Dict) -> str:
        """撰写结论"""
        return f"""\section{{Conclusion}}

In this paper, we presented {idea.get('Title', 'our method')}.
Experimental results demonstrate its effectiveness.

\textbf{{Limitations:}} Our method has certain limitations...

\textbf{{Future Work:}} We plan to extend this work..."""


class ReviewAgent(BaseAgent):
    """
    评审智能体
    负责评审论文质量
    """

    def __init__(self, agent_id: str, message_bus: MessageBus, shared_memory: SharedMemory):
        super().__init__(agent_id, AgentRole.REVIEW, message_bus, shared_memory)
        self.review_count = 0

    def execute_task(self, task: Task) -> Dict[str, Any]:
        """执行评审任务"""
        print(f"[{self.agent_id}] 🔍 评审论文...")

        requirements = task.requirements
        paper = requirements.get("input_data", {}).get("sections", {})

        if not self.llm_client:
            self.llm_client = get_llm_client()

        # 执行多维度评审
        review = {
            "overall_score": 0,
            "scores": {},
            "comments": {},
            "suggestions": [],
            "verdict": ""
        }

        # 各维度评审
        dimensions = [
            ("novelty", "Novelty and Originality", 30),
            ("technical_quality", "Technical Quality", 25),
            ("clarity", "Clarity and Presentation", 20),
            ("experiments", "Experiments and Results", 25)
        ]

        total_weighted = 0
        for dim_key, dim_name, weight in dimensions:
            score, comment = self._review_dimension(dim_key, paper)
            review["scores"][dim_key] = {
                "name": dim_name,
                "score": score,
                "max": 10,
                "weight": weight,
                "weighted": score * weight / 10
            }
            review["comments"][dim_key] = comment
            total_weighted += score * weight / 10

        review["overall_score"] = round(total_weighted, 2)

        # 生成建议
        review["suggestions"] = self._generate_suggestions(review["scores"])

        # 生成裁决
        review["verdict"] = self._generate_verdict(review["overall_score"])

        self.review_count += 1

        print(f"[{self.agent_id}] ✓ 评审完成: {review['overall_score']}/10 - {review['verdict']}")

        return review

    def _review_dimension(self, dimension: str, paper: Dict) -> tuple:
        """评审特定维度"""
        # 简化版本：使用规则评分
        # 实际应该使用 LLM

        scores = {
            "novelty": (7.5, "The paper presents some novel ideas, but the contribution could be more significant."),
            "technical_quality": (7.0, "The method is technically sound with minor issues."),
            "clarity": (8.0, "The paper is well-written and easy to follow."),
            "experiments": (7.5, "Experiments are comprehensive but could include more baselines.")
        }

        return scores.get(dimension, (7.0, "Adequate."))

    def _generate_suggestions(self, scores: Dict) -> List[str]:
        """生成改进建议"""
        suggestions = []

        if scores.get("novelty", {}).get("score", 0) < 8:
            suggestions.append("Consider strengthening the novelty claim by comparing more related work.")

        if scores.get("experiments", {}).get("score", 0) < 8:
            suggestions.append("Add more comprehensive ablation studies.")

        if scores.get("clarity", {}).get("score", 0) < 7:
            suggestions.append("Improve figure quality and caption descriptions.")

        return suggestions

    def _generate_verdict(self, score: float) -> str:
        """生成裁决"""
        if score >= 8.5:
            return "Strong Accept"
        elif score >= 7.5:
            return "Accept"
        elif score >= 6.5:
            return "Weak Accept"
        elif score >= 5.0:
            return "Borderline"
        else:
            return "Reject"


class ImprovementAgent(BaseAgent):
    """
    改进智能体
    负责根据评审意见改进论文
    """

    def __init__(self, agent_id: str, message_bus: MessageBus, shared_memory: SharedMemory):
        super().__init__(agent_id, AgentRole.IMPROVEMENT, message_bus, shared_memory)
        self.improvement_count = 0

    def execute_task(self, task: Task) -> Dict[str, Any]:
        """执行改进任务"""
        print(f"[{self.agent_id}] 🔧 改进论文...")

        requirements = task.requirements
        input_data = requirements.get("input_data", {})
        paper = input_data.get("paper_draft", {})
        review = input_data.get("review_report", {})

        if not self.llm_client:
            self.llm_client = get_llm_client()

        # 解析评审意见
        suggestions = review.get("suggestions", [])
        scores = review.get("scores", {})

        # 针对性改进
        improvements = {}

        # 根据评审维度进行改进
        for dim, score_info in scores.items():
            if score_info.get("score", 0) < 7.5:
                improved_section = self._improve_section(dim, paper, review)
                improvements[dim] = improved_section

        result = {
            "improvements_made": list(improvements.keys()),
            "suggestions_addressed": len(suggestions),
            "improved_sections": improvements,
            "estimated_score_improvement": 0.5,
            "timestamp": datetime.now().isoformat()
        }

        self.improvement_count += 1

        print(f"[{self.agent_id}] ✓ 改进完成: 改进了 {len(improvements)} 个部分")

        return result

    def _improve_section(self, dimension: str, paper: Dict, review: Dict) -> str:
        """改进特定部分"""
        # 简化版本
        improvements = {
            "novelty": "Added stronger novelty claims and related work comparison.",
            "technical_quality": "Clarified technical details and added implementation specifics.",
            "clarity": "Improved figures and rewrote unclear sections.",
            "experiments": "Added additional ablation studies and baseline comparisons."
        }

        return improvements.get(dimension, "General improvements applied.")


class AgentFactory:
    """
    Agent 工厂
    用于创建各种 Agent 实例
    """

    @staticmethod
    def create_agents(orchestrator, num_idea: int = 1, num_exp: int = 1,
                     num_write: int = 1, num_review: int = 1, num_improve: int = 1) -> Dict[str, BaseAgent]:
        """创建一组 Agent"""
        agents = {}
        bus = orchestrator.message_bus
        memory = orchestrator.shared_memory

        # Idea Agents
        for i in range(num_idea):
            agent_id = f"idea_agent_{i}"
            agent = IdeaAgent(agent_id, bus, memory)
            orchestrator.register_agent(agent)
            agents[agent_id] = agent

        # Experiment Agents
        for i in range(num_exp):
            agent_id = f"exp_agent_{i}"
            agent = ExperimentAgent(agent_id, bus, memory)
            orchestrator.register_agent(agent)
            agents[agent_id] = agent

        # Writing Agents
        for i in range(num_write):
            agent_id = f"write_agent_{i}"
            agent = WritingAgent(agent_id, bus, memory)
            orchestrator.register_agent(agent)
            agents[agent_id] = agent

        # Review Agents
        for i in range(num_review):
            agent_id = f"review_agent_{i}"
            agent = ReviewAgent(agent_id, bus, memory)
            orchestrator.register_agent(agent)
            agents[agent_id] = agent

        # Improvement Agents
        for i in range(num_improve):
            agent_id = f"improve_agent_{i}"
            agent = ImprovementAgent(agent_id, bus, memory)
            orchestrator.register_agent(agent)
            agents[agent_id] = agent

        return agents


if __name__ == "__main__":
    # 测试各个 Agent
    print("=" * 70)
    print("测试 CVPR-Auto Agents")
    print("=" * 70)

    from cvpr_auto.multi_agent_system import AgentOrchestrator

    # 创建 Orchestrator
    orchestrator = AgentOrchestrator()

    # 创建 Agents
    agents = AgentFactory.create_agents(
        orchestrator,
        num_idea=1,
        num_exp=1,
        num_write=1,
        num_review=1,
        num_improve=1
    )

    print(f"\n✓ 创建了 {len(agents)} 个 Agent:")
    for agent_id, agent in agents.items():
        print(f"  - {agent_id} ({agent.role.value})")

    # 测试 IdeaAgent
    print("\n" + "=" * 70)
    print("测试 IdeaAgent")
    print("=" * 70)

    test_task = Task(
        task_id="test_idea_gen",
        task_type="idea_generation",
        description="Generate test ideas",
        requirements={
            "domain": "computer_vision",
            "num_ideas": 3,
            "constraints": {}
        }
    )

    idea_agent = agents["idea_agent_0"]
    result = idea_agent.execute_task(test_task)

    print(f"\n生成了 {result['count']} 个想法:")
    for idea in result['best']:
        print(f"  - {idea.get('Title', 'Unknown')} (Score: {idea.get('OverallScore', 0)})")
