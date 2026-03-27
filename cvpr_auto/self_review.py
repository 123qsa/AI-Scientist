"""
自评审模块
模拟审稿人视角进行多轮改进
"""

from typing import Dict, List, Optional
import json
from dataclasses import dataclass


@dataclass
class ReviewScore:
    """评审分数"""
    novelty: float  # 1-10
    significance: float
    technical_quality: float
    clarity: float
    reproducibility: float
    overall: float

    def to_dict(self):
        return {
            'novelty': self.novelty,
            'significance': self.significance,
            'technical_quality': self.technical_quality,
            'clarity': self.clarity,
            'reproducibility': self.reproducibility,
            'overall': self.overall
        }


class SelfReviewer:
    """自评审器"""

    def __init__(self, quality_thresholds: Dict):
        self.thresholds = quality_thresholds
        self.review_history = []

    def review_paper(self, paper_content: Dict, experiment_results: Dict) -> Dict:
        """
        评审论文

        返回评审意见和改进建议
        """
        review = {
            'scores': {},
            'strengths': [],
            'weaknesses': [],
            'questions': [],
            'suggestions': []
        }

        # 1. 评估创新性
        novelty_score = self._evaluate_novelty(paper_content, experiment_results)
        review['scores']['novelty'] = novelty_score

        if novelty_score < self.thresholds['novelty_score']:
            review['weaknesses'].append(
                f"Novelty score {novelty_score:.1f} below threshold {self.thresholds['novelty_score']}"
            )
            review['suggestions'].append(
                "Consider adding theoretical analysis or novel architectural insights"
            )
        else:
            review['strengths'].append("The proposed method shows sufficient novelty")

        # 2. 评估实验质量
        exp_score = self._evaluate_experiments(experiment_results)
        review['scores']['experiment_rigor'] = exp_score

        if exp_score < self.thresholds['experiment_rigor']:
            review['weaknesses'].append("Experimental evaluation is insufficient")
            if 'ablation' not in experiment_results or \
               len(experiment_results.get('ablation', {})) < self.thresholds['min_ablations']:
                review['suggestions'].append(
                    f"Add more ablation studies (need at least {self.thresholds['min_ablations']})"
                )
            if len(experiment_results.get('datasets', [])) < self.thresholds['min_datasets']:
                review['suggestions'].append(
                    f"Evaluate on more datasets (need at least {self.thresholds['min_datasets']})"
                )

        # 3. 评估写作
        writing_score = self._evaluate_writing(paper_content)
        review['scores']['writing_quality'] = writing_score

        if writing_score < self.thresholds['writing_quality']:
            review['weaknesses'].append("Paper writing needs improvement")
            review['suggestions'].append("Consider professional proofreading or more revision rounds")

        # 4. 评估重要性
        significance = self._evaluate_significance(experiment_results)
        review['scores']['significance'] = significance

        improvement = experiment_results.get('improvement_over_sota', 0)
        if improvement < self.thresholds['min_improvement']:
            review['weaknesses'].append(
                f"Performance improvement ({improvement:.2f}%) is marginal"
            )
            review['suggestions'].append(
                f"Target at least {self.thresholds['min_improvement']}% improvement over SOTA"
            )

        # 5. 计算总体分数
        overall = sum(review['scores'].values()) / len(review['scores'])
        review['scores']['overall'] = round(overall, 2)

        # 6. 生成审稿问题
        review['questions'] = self._generate_review_questions(paper_content, experiment_results)

        self.review_history.append(review)

        return review

    def _evaluate_novelty(self, paper_content: Dict, results: Dict) -> float:
        """评估创新性"""
        score = 5.0

        # 检查是否有理论贡献
        if 'theoretical' in paper_content.get('method', '').lower():
            score += 1.0

        # 检查是否解决明确的问题
        if 'problem' in paper_content.get('introduction', '').lower():
            score += 0.5

        # 检查复杂度分析
        if 'complexity' in paper_content.get('method', '').lower():
            score += 0.5

        # 检查实验提升幅度
        improvement = results.get('improvement_over_sota', 0)
        if improvement > 2.0:
            score += 2.0
        elif improvement > 1.0:
            score += 1.0

        return min(score, 10.0)

    def _evaluate_experiments(self, results: Dict) -> float:
        """评估实验质量"""
        score = 5.0

        # 数据集数量
        num_datasets = len(results.get('datasets', []))
        if num_datasets >= 3:
            score += 2.0
        elif num_datasets >= 2:
            score += 1.0

        # 消融实验数量
        num_ablations = len(results.get('ablation', {}))
        if num_ablations >= 5:
            score += 2.0
        elif num_ablations >= 3:
            score += 1.0

        # 是否有超参分析
        if 'hyperparam_search' in results:
            score += 0.5

        # 是否有可视化
        if 'visualizations' in results:
            score += 0.5

        return min(score, 10.0)

    def _evaluate_writing(self, paper_content: Dict) -> float:
        """评估写作质量"""
        score = 6.0

        # 检查各章节长度
        for section in ['introduction', 'method', 'experiments']:
            text = paper_content.get(section, '')
            words = len(text.split())

            if section == 'introduction' and words > 500:
                score += 0.5
            elif section == 'method' and words > 800:
                score += 0.5
            elif section == 'experiments' and words > 600:
                score += 0.5

        # 检查是否有清晰的贡献列表
        if 'contribution' in paper_content.get('introduction', '').lower():
            score += 0.5

        return min(score, 10.0)

    def _evaluate_significance(self, results: Dict) -> float:
        """评估重要性"""
        score = 5.0

        improvement = results.get('improvement_over_sota', 0)

        if improvement > 3.0:
            score += 4.0
        elif improvement > 2.0:
            score += 3.0
        elif improvement > 1.0:
            score += 1.5

        # 是否开源
        if results.get('code_available', False):
            score += 0.5

        # 是否在多个任务有效
        if len(results.get('datasets', [])) >= 3:
            score += 0.5

        return min(score, 10.0)

    def _generate_review_questions(self, paper_content: Dict, results: Dict) -> List[str]:
        """生成审稿问题"""
        questions = []

        # 基于内容生成问题
        method = paper_content.get('method', '')

        if 'attention' in method.lower():
            questions.append("How does the proposed attention mechanism compare to standard self-attention in terms of computational complexity?")

        if 'loss' in method.lower():
            questions.append("Can you provide more intuition about why the proposed loss function is better suited for this task?")

        if len(results.get('datasets', [])) < 3:
            questions.append("Have you tested the method on other datasets to verify generalization?")

        if 'ablation' in results:
            ablation = results['ablation']
            if len(ablation) < 5:
                questions.append("The ablation study seems limited. Can you include ablations on [specific component]?")

        questions.append("What are the limitations of the proposed method? Please discuss failure cases.")

        return questions

    def should_continue_iteration(self, review: Dict, max_rounds: int = 5) -> bool:
        """判断是否应继续迭代"""
        # 如果已经达到阈值，停止
        if review['scores']['overall'] >= 7.5:
            return False

        # 如果已达到最大轮数，停止
        if len(self.review_history) >= max_rounds:
            return False

        # 如果本轮没有改进，停止
        if len(self.review_history) >= 2:
            prev_score = self.review_history[-2]['scores']['overall']
            curr_score = review['scores']['overall']
            if curr_score <= prev_score:
                return False

        return True

    def generate_improvement_prompt(self, review: Dict) -> str:
        """生成改进提示"""
        prompt = f"""
Based on the self-review of the current paper (Overall score: {review['scores']['overall']:.1f}/10), here are the key issues to address:

Weaknesses:
"""
        for w in review['weaknesses']:
            prompt += f"- {w}\n"

        prompt += "\nSuggested Improvements:\n"
        for s in review['suggestions']:
            prompt += f"- {s}\n"

        prompt += "\nReview Questions to Address:\n"
        for q in review['questions']:
            prompt += f"- {q}\n"

        prompt += """
Please revise the paper to address these issues. Focus on:
1. Strengthening the novelty claim with theoretical or empirical evidence
2. Adding missing experiments or analysis
3. Improving clarity and presentation
4. Addressing the review questions directly in the text
"""

        return prompt


class QualityGate:
    """质量关卡"""

    def __init__(self, thresholds: Dict):
        self.thresholds = thresholds

    def check(self, review: Dict) -> Dict:
        """检查是否通过质量关卡"""
        results = {
            'passed': True,
            'failed_checks': [],
            'recommendations': []
        }

        checks = [
            ('novelty_score', review['scores']['novelty'], 'Novelty'),
            ('experiment_rigor', review['scores']['experiment_rigor'], 'Experiment Rigor'),
            ('writing_quality', review['scores']['writing_quality'], 'Writing Quality'),
            ('significance', review['scores']['significance'], 'Significance'),
        ]

        for key, value, name in checks:
            threshold = self.thresholds.get(key, 7.0)
            if value < threshold:
                results['passed'] = False
                results['failed_checks'].append(
                    f"{name}: {value:.1f} < {threshold}"
                )

        # 检查实验数量
        # (实际应该在 experiment_results 中检查)

        if not results['passed']:
            results['recommendations'].append(
                "Paper does not meet CVPR quality threshold. "
                "Consider major revision or targeting a workshop."
            )
        else:
            results['recommendations'].append(
                "Paper meets minimum quality threshold for CVPR submission. "
                "Proceed with professional review."
            )

        return results


if __name__ == '__main__':
    # 示例使用
    thresholds = {
        'novelty_score': 7.5,
        'experiment_rigor': 8.0,
        'writing_quality': 7.5,
        'significance': 7.0,
        'min_improvement': 1.0,
        'min_ablations': 5,
        'min_datasets': 2,
    }

    reviewer = SelfReviewer(thresholds)

    paper = {
        'title': 'Test Paper',
        'abstract': 'This is a test.',
        'introduction': 'Deep learning is important. Our contributions are: 1)... 2)...',
        'method': 'Our method uses attention mechanism with O(n) complexity.',
        'experiments': 'We test on ImageNet.'
    }

    results = {
        'improvement_over_sota': 1.5,
        'datasets': ['imagenet'],
        'ablation': {'full': {}, 'w/o_att': {}}
    }

    review = reviewer.review_paper(paper, results)
    print(json.dumps(review, indent=2))
