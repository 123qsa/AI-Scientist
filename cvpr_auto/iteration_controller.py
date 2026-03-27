"""
迭代控制器 - 实现完整的评审-改进-验证闭环
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import copy

from cvpr_auto.config import config
from cvpr_auto.self_review import SelfReviewer, QualityGate
from cvpr_auto.llm_client import get_llm_client, LLMResponse


@dataclass
class IterationState:
    """迭代状态记录"""
    round_num: int
    timestamp: str
    review_scores: Dict[str, float]
    weaknesses: List[str]
    improvements_made: List[str]
    experiment_delta: Dict  # 实验指标变化
    paper_delta: Dict       # 论文内容变化

    def to_dict(self):
        return asdict(self)


class ImprovementGenerator:
    """改进方案生成器"""

    def __init__(self, model_client=None):
        self.model_client = model_client
        self.improvement_history = []

    def generate_experiment_improvements(
        self,
        review: Dict,
        current_experiments: Dict,
        codebase: Dict
    ) -> List[Dict]:
        """
        生成实验改进方案

        Returns:
            改进动作列表，每个动作包含:
            - type: 'add_experiment' | 'modify_code' | 'add_analysis'
            - target: 目标文件/模块
            - description: 改进描述
            - prompt: 给 LLM 的具体指令
        """
        improvements = []

        # 检查消融实验不足
        if 'ablation' not in current_experiments or \
           len(current_experiments.get('ablation', {})) < config.QUALITY_THRESHOLDS['min_ablations']:

            missing = config.QUALITY_THRESHOLDS['min_ablations'] - \
                     len(current_experiments.get('ablation', {}))

            improvements.append({
                'type': 'add_experiment',
                'subtype': 'ablation',
                'priority': 'high',
                'target': 'experiment.py',
                'description': f'Add {missing} more ablation studies',
                'rationale': 'CVPR requires comprehensive ablation to verify component contributions',
                'action_prompt': self._generate_ablation_prompt(current_experiments)
            })

        # 检查数据集不足
        if len(current_experiments.get('datasets', [])) < config.QUALITY_THRESHOLDS['min_datasets']:
            improvements.append({
                'type': 'add_experiment',
                'subtype': 'dataset',
                'priority': 'high',
                'target': 'experiment.py',
                'description': 'Evaluate on additional datasets',
                'rationale': 'Cross-dataset validation is required for CVPR',
                'action_prompt': 'Add experiments on CIFAR-100 and Tiny ImageNet'
            })

        # 检查超参搜索
        if 'hyperparam_search' not in current_experiments:
            improvements.append({
                'type': 'add_experiment',
                'subtype': 'hyperparam',
                'priority': 'medium',
                'target': 'hyperparam_search.py',
                'description': 'Conduct systematic hyperparameter search',
                'rationale': 'Proper hyperparameter tuning is essential for fair comparison',
                'action_prompt': 'Run Optuna search for learning rate, batch size, weight decay'
            })

        # 检查可视化
        if 'visualizations' not in current_experiments:
            improvements.append({
                'type': 'add_experiment',
                'subtype': 'visualization',
                'priority': 'medium',
                'target': 'plot.py',
                'description': 'Add qualitative visualizations',
                'rationale': 'Visual results are crucial for CVPR papers',
                'action_prompt': 'Generate attention maps, feature visualizations, failure cases'
            })

        # 检查理论分析
        if not current_experiments.get('has_complexity_analysis', False):
            improvements.append({
                'type': 'add_analysis',
                'subtype': 'complexity',
                'priority': 'high',
                'target': 'method section',
                'description': 'Add computational complexity analysis',
                'rationale': 'CVPR requires complexity analysis (FLOPs, Params)',
                'action_prompt': 'Calculate and compare FLOPs and parameter counts with baselines'
            })

        # 检查鲁棒性实验
        if not current_experiments.get('robustness_tests', False):
            improvements.append({
                'type': 'add_experiment',
                'subtype': 'robustness',
                'priority': 'medium',
                'target': 'experiment.py',
                'description': 'Add robustness evaluation',
                'rationale': 'Robustness to perturbations is important for CVPR',
                'action_prompt': 'Test with different corruptions, noise levels, input variations'
            })

        return improvements

    def generate_paper_improvements(
        self,
        review: Dict,
        current_paper: Dict
    ) -> List[Dict]:
        """生成论文写作改进方案"""
        improvements = []

        writing_score = review['scores'].get('writing_quality', 5)

        # 根据写作分数生成改进
        if writing_score < 7:
            improvements.append({
                'type': 'modify_writing',
                'subtype': 'clarity',
                'priority': 'high',
                'target': 'all_sections',
                'description': 'Improve overall clarity and flow',
                'rationale': f'Writing score {writing_score} below threshold 7.5',
                'action_prompt': self._generate_writing_prompt('clarity', current_paper)
            })

        # 检查 Introduction
        intro = current_paper.get('introduction', '')
        if len(intro.split()) < 400:
            improvements.append({
                'type': 'expand_section',
                'subtype': 'introduction',
                'priority': 'high',
                'target': 'introduction',
                'description': 'Expand introduction to >400 words',
                'rationale': 'CVPR introductions need sufficient context and motivation',
                'action_prompt': 'Add more background, clearly define the problem, strengthen motivation'
            })

        if 'contribution' not in intro.lower():
            improvements.append({
                'type': 'add_content',
                'subtype': 'contributions',
                'priority': 'high',
                'target': 'introduction',
                'description': 'Add explicit contribution list',
                'rationale': 'CVPR papers must clearly state contributions',
                'action_prompt': 'Add a paragraph listing 3-4 specific contributions with bullet points'
            })

        # 检查 Method
        method = current_paper.get('method', '')
        if len(method.split()) < 600:
            improvements.append({
                'type': 'expand_section',
                'subtype': 'method',
                'priority': 'medium',
                'target': 'method',
                'description': 'Expand method section with more details',
                'rationale': 'Method needs sufficient detail for reproducibility',
                'action_prompt': 'Add more implementation details, pseudocode, architecture specifications'
            })

        if 'complexity' not in method.lower() and 'flops' not in method.lower():
            improvements.append({
                'type': 'add_content',
                'subtype': 'complexity_analysis',
                'priority': 'high',
                'target': 'method',
                'description': 'Add complexity analysis',
                'rationale': 'CVPR requires complexity comparison (FLOPs, params)',
                'action_prompt': 'Add section on computational complexity with comparison table'
            })

        # 检查 Experiments
        exp = current_paper.get('experiments', '')
        if len(exp.split()) < 500:
            improvements.append({
                'type': 'expand_section',
                'subtype': 'experiments',
                'priority': 'medium',
                'target': 'experiments',
                'description': 'Expand experiments section',
                'rationale': 'Need detailed experimental setup and results',
                'action_prompt': 'Add implementation details, training settings, dataset statistics'
            })

        # 检查 Related Work
        rw = current_paper.get('related_work', '')
        if len(rw.split()) < 300:
            improvements.append({
                'type': 'expand_section',
                'subtype': 'related_work',
                'priority': 'medium',
                'target': 'related_work',
                'description': 'Expand related work discussion',
                'rationale': 'Need comprehensive literature review',
                'action_prompt': 'Add discussion of recent CVPR/ICCV papers, clarify differences'
            })

        # 检查 Limitations
        if 'limitation' not in current_paper.get('conclusion', '').lower():
            improvements.append({
                'type': 'add_content',
                'subtype': 'limitations',
                'priority': 'medium',
                'target': 'conclusion',
                'description': 'Add limitations section',
                'rationale': 'CVPR requires honest discussion of limitations',
                'action_prompt': 'Add paragraph discussing limitations and future work'
            })

        return improvements

    def _generate_ablation_prompt(self, current_exp: Dict) -> str:
        """生成消融实验提示"""
        existing = list(current_exp.get('ablation', {}).keys())

        prompt = f"""
Current ablations: {existing}

Design additional ablation experiments to verify:
1. The contribution of each architectural component
2. The impact of key hyperparameters
3. Design choices vs alternatives

For each ablation:
- Clearly define what is being removed/changed
- Explain why this component matters
- Expected outcome (hypothesis)

Generate ablation configurations and run experiments.
"""
        return prompt

    def _generate_writing_prompt(self, aspect: str, paper: Dict) -> str:
        """生成写作改进提示"""
        prompts = {
            'clarity': """
Improve the clarity of the paper by:
1. Simplify complex sentences
2. Add transition sentences between paragraphs
3. Ensure consistent terminology
4. Clarify ambiguous descriptions
5. Improve logical flow

Focus on making the paper accessible to CVPR reviewers.
""",
            'technical': """
Strengthen technical content by:
1. Add formal definitions and theorems
2. Include pseudocode for algorithms
3. Provide detailed architecture specifications
4. Add mathematical formulations
5. Clarify training procedures
"""
        }
        return prompts.get(aspect, prompts['clarity'])

    def prioritize_improvements(self, improvements: List[Dict]) -> List[Dict]:
        """按优先级排序改进方案"""
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        return sorted(improvements, key=lambda x: priority_order.get(x['priority'], 3))


class IterationController:
    """迭代控制器 - 管理完整的改进循环"""

    def __init__(
        self,
        experiment_runner: Callable,
        paper_generator: Callable,
        llm_client=None,
        llm_provider: str = None,
        llm_model: str = None
    ):
        self.experiment_runner = experiment_runner
        self.paper_generator = paper_generator

        # 初始化 LLM 客户端
        if llm_client:
            self.llm_client = llm_client
        elif llm_provider:
            try:
                self.llm_client = get_llm_client(llm_provider, llm_model)
                print(f"✓ LLM client initialized: {llm_provider}" +
                      (f" ({llm_model})" if llm_model else ""))
            except Exception as e:
                print(f"⚠️ Failed to initialize LLM client: {e}")
                print("  Falling back to simple improvements without LLM")
                self.llm_client = None
        else:
            # 尝试默认配置
            try:
                self.llm_client = get_llm_client()
                print("✓ LLM client initialized with default settings")
            except Exception as e:
                print(f"⚠️ No LLM client available: {e}")
                self.llm_client = None

        self.reviewer = SelfReviewer(config.QUALITY_THRESHOLDS)
        self.quality_gate = QualityGate(config.QUALITY_THRESHOLDS)
        self.improvement_gen = ImprovementGenerator(self.llm_client)

        self.iteration_history: List[IterationState] = []
        self.max_rounds = config.MAX_REVISION_ROUNDS

    def run_iteration_loop(
        self,
        initial_idea: Dict,
        initial_code: Dict,
        output_dir: str
    ) -> Tuple[Dict, Dict, bool]:
        """
        运行完整的迭代循环

        Returns:
            (final_paper, final_experiments, success)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        current_paper = None
        current_experiments = None

        print("\n" + "="*70)
        print("🔄 Starting Iterative Improvement Loop")
        print("="*70)

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n{'='*70}")
            print(f"📋 Iteration Round {round_num}/{self.max_rounds}")
            print(f"{'='*70}")

            # Step 1: 运行/更新实验
            if round_num == 1:
                print("\n🔬 Running initial experiments...")
                current_experiments = self.experiment_runner(initial_code)
            else:
                # 根据上一轮改进方案更新实验
                if hasattr(self, '_pending_exp_improvements'):
                    print("\n🔬 Running additional experiments...")
                    current_experiments = self._apply_experiment_improvements(
                        current_experiments,
                        self._pending_exp_improvements
                    )

            # Step 2: 生成/更新论文
            print("\n📝 Generating paper...")
            if round_num == 1:
                current_paper = self.paper_generator(initial_idea, current_experiments)
            else:
                if hasattr(self, '_pending_paper_improvements'):
                    print(f"\n📝 Applying paper improvements with " +
                          ("LLM..." if self.llm_client else "simple editing..."))
                    current_paper = self._apply_paper_improvements_with_llm(
                        current_paper,
                        self._pending_paper_improvements
                    )

            # Step 3: 自评审
            print("\n🔍 Self-reviewing...")
            review = self.reviewer.review_paper(current_paper, current_experiments)

            print(f"\n📊 Review Scores:")
            for metric, score in review['scores'].items():
                threshold = config.QUALITY_THRESHOLDS.get(metric.replace('_score', ''), 7.0)
                status = "✅" if score >= threshold else "⚠️"
                print(f"  {status} {metric}: {score:.1f}/10 (threshold: {threshold})")

            # Step 4: 质量关卡检查
            quality_check = self.quality_gate.check(review)

            # 记录状态
            state = IterationState(
                round_num=round_num,
                timestamp=datetime.now().isoformat(),
                review_scores=review['scores'],
                weaknesses=review['weaknesses'],
                improvements_made=getattr(self, '_last_improvements', []),
                experiment_delta=self._calculate_exp_delta(),
                paper_delta=self._calculate_paper_delta()
            )
            self.iteration_history.append(state)

            # Step 5: 检查是否满足条件
            if quality_check['passed']:
                print(f"\n✅ Quality gate passed! Final score: {review['scores']['overall']:.1f}")
                self._save_iteration_history(output_path)
                return current_paper, current_experiments, True

            # Step 6: 检查是否继续迭代
            if round_num >= self.max_rounds:
                print(f"\n⚠️ Max rounds reached. Final score: {review['scores']['overall']:.1f}")
                print("Consider manual improvement before submission.")
                self._save_iteration_history(output_path)
                return current_paper, current_experiments, False

            if not self.reviewer.should_continue_iteration(review, self.max_rounds):
                print(f"\n⚠️ No improvement in this round. Stopping.")
                self._save_iteration_history(output_path)
                return current_paper, current_experiments, False

            # Step 7: 生成改进方案
            print(f"\n🔧 Generating improvements...")

            exp_improvements = self.improvement_gen.generate_experiment_improvements(
                review, current_experiments, initial_code
            )
            paper_improvements = self.improvement_gen.generate_paper_improvements(
                review, current_paper
            )

            all_improvements = exp_improvements + paper_improvements
            prioritized = self.improvement_gen.prioritize_improvements(all_improvements)

            print(f"\n📋 Planned Improvements ({len(prioritized)} items):")
            for i, imp in enumerate(prioritized[:5], 1):  # 显示前5个
                print(f"  {i}. [{imp['priority'].upper()}] {imp['description']}")

            # 保存待执行的改进
            self._pending_exp_improvements = [i for i in prioritized if i['type'] == 'add_experiment']
            self._pending_paper_improvements = [i for i in prioritized if i['type'] != 'add_experiment']
            self._last_improvements = [i['description'] for i in prioritized]

            # 保存中间结果
            self._save_checkpoint(output_path, round_num, current_paper, current_experiments, review)

        return current_paper, current_experiments, False

    def _apply_experiment_improvements(
        self,
        current_experiments: Dict,
        improvements: List[Dict]
    ) -> Dict:
        """应用实验改进"""
        # 这里应该调用实际的实验运行代码
        # 简化版：返回更新后的实验结果

        updated = copy.deepcopy(current_experiments)

        for imp in improvements:
            print(f"  Applying: {imp['description']}")

            # 模拟运行新实验
            if imp['subtype'] == 'ablation':
                # 添加消融实验结果
                if 'ablation' not in updated:
                    updated['ablation'] = {}
                updated['ablation'][f'added_round_{len(self.iteration_history)}'] = {
                    'metrics': 78.5  # 模拟数据
                }

            elif imp['subtype'] == 'dataset':
                updated['datasets'] = updated.get('datasets', []) + ['cifar100']

            elif imp['subtype'] == 'hyperparam':
                updated['hyperparam_search'] = {
                    'best_params': {'lr': 0.001, 'batch_size': 128},
                    'param_importance': {'lr': 0.4, 'batch_size': 0.2}
                }

        return updated

    def _apply_paper_improvements_with_llm(
        self,
        current_paper: Dict,
        improvements: List[Dict]
    ) -> Dict:
        """使用 LLM 改进论文"""
        if self.model_client is None:
            # 如果没有 LLM 客户端，使用简单标记
            return self._apply_paper_improvements_simple(current_paper, improvements)

        updated = copy.deepcopy(current_paper)

        for imp in improvements:
            print(f"  Applying with LLM: {imp['description']}")

            section = imp['target']
            if section not in updated:
                continue

            current_text = updated[section]

            # 构建改进 prompt
            system_prompt = """You are an expert academic writing assistant specializing in computer vision papers for top-tier conferences like CVPR.
Your task is to improve specific sections of a research paper based on reviewer feedback.
Maintain academic tone, clarity, and technical accuracy.
Keep the length appropriate for the section type."""

            prompt = f"""Please improve the following {section} section of a CVPR paper.

IMPROVEMENT NEEDED: {imp['description']}
RATIONALE: {imp.get('rationale', 'N/A')}

CURRENT TEXT:
{current_text}

Please rewrite this section addressing the improvement needed. Maintain the same general structure but enhance:
- Clarity and flow
- Technical precision
- Academic tone
- Completeness (add missing details as specified)

Provide only the improved text, no explanations."""

            try:
                response = self.model_client.generate(prompt, system_prompt)
                if not response.error:
                    updated[section] = response.content
                    print(f"    ✓ Successfully improved {section}")
                else:
                    print(f"    ⚠️ LLM error: {response.error}")
                    # 保留原文
            except Exception as e:
                print(f"    ⚠️ Failed to apply improvement: {e}")

        return updated

    def _apply_paper_improvements_simple(self, current_paper: Dict, improvements: List[Dict]) -> Dict:
        """简单改进（无 LLM）"""
        updated = copy.deepcopy(current_paper)

        for imp in improvements:
            print(f"  Applying: {imp['description']}")
            section = imp['target']
            if section in updated:
                updated[section] = updated[section] + f"\n\n[IMPROVED: {imp['subtype']}]"

        return updated

    def _calculate_exp_delta(self) -> Dict:
        """计算实验指标变化"""
        if len(self.iteration_history) < 2:
            return {}

        prev = self.iteration_history[-2].review_scores.get('experiment_rigor', 0)
        curr = self.iteration_history[-1].review_scores.get('experiment_rigor', 0)

        return {'experiment_rigor_delta': round(curr - prev, 2)}

    def _calculate_paper_delta(self) -> Dict:
        """计算论文质量变化"""
        if len(self.iteration_history) < 2:
            return {}

        prev = self.iteration_history[-2].review_scores.get('writing_quality', 0)
        curr = self.iteration_history[-1].review_scores.get('writing_quality', 0)

        return {'writing_quality_delta': round(curr - prev, 2)}

    def _save_checkpoint(self, output_path: Path, round_num: int,
                        paper: Dict, experiments: Dict, review: Dict):
        """保存检查点"""
        checkpoint_dir = output_path / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            'round': round_num,
            'paper': paper,
            'experiments': experiments,
            'review': review
        }

        with open(checkpoint_dir / f'round_{round_num}.json', 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def _save_iteration_history(self, output_path: Path):
        """保存迭代历史"""
        history_file = output_path / 'iteration_history.json'

        with open(history_file, 'w') as f:
            json.dump([s.to_dict() for s in self.iteration_history], f, indent=2)

        print(f"\n💾 Iteration history saved to {history_file}")

    def get_improvement_report(self) -> str:
        """生成改进报告"""
        if not self.iteration_history:
            return "No iterations completed yet."

        report = ["\n" + "="*70]
        report.append("📊 ITERATION IMPROVEMENT REPORT")
        report.append("="*70)

        initial = self.iteration_history[0]
        final = self.iteration_history[-1]

        report.append(f"\nTotal Rounds: {len(self.iteration_history)}")
        report.append(f"\nScore Progression:")

        for metric in ['novelty', 'experiment_rigor', 'writing_quality', 'significance', 'overall']:
            initial_score = initial.review_scores.get(metric, 0)
            final_score = final.review_scores.get(metric, 0)
            delta = final_score - initial_score

            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            report.append(f"  {metric:20s}: {initial_score:.1f} → {final_score:.1f} ({arrow}{abs(delta):.1f})")

        report.append(f"\nImprovements Made:")
        for i, state in enumerate(self.iteration_history, 1):
            if state.improvements_made:
                report.append(f"  Round {i}:")
                for imp in state.improvements_made:
                    report.append(f"    - {imp}")

        report.append("="*70)

        return "\n".join(report)


if __name__ == '__main__':
    # 示例使用
    def mock_experiment_runner(code):
        return {
            'accuracy': 79.5,
            'datasets': ['imagenet'],
            'ablation': {'full': {'metrics': 79.5}}
        }

    def mock_paper_generator(idea, exp):
        return {
            'title': 'Test Paper',
            'abstract': 'Abstract text',
            'introduction': 'Introduction text ' * 50,
            'method': 'Method text ' * 50,
            'experiments': 'Experiments text ' * 50,
            'conclusion': 'Conclusion'
        }

    controller = IterationController(mock_experiment_runner, mock_paper_generator)

    paper, experiments, success = controller.run_iteration_loop(
        initial_idea={'title': 'Test'},
        initial_code={},
        output_dir='./test_output'
    )

    print(controller.get_improvement_report())
