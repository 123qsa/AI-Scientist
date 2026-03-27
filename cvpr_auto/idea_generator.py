"""
想法生成与新颖性检查模块
自动生成研究想法并验证其新颖性
"""

import json
import os
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import random

from cvpr_auto.llm_client import get_llm_client, LLMResponse
from cvpr_auto.config import config


@dataclass
class ResearchIdea:
    """研究想法数据结构"""
    id: str
    title: str
    abstract: str
    problem_statement: str
    proposed_solution: str
    expected_contributions: List[str]
    key_innovations: List[str]
    methodology: str
    novelty_score: float = 0.0
    relevance_score: float = 0.0
    feasibility_score: float = 0.0
    similar_papers: List[Dict] = None
    is_novel: bool = True

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'title': self.title,
            'abstract': self.abstract,
            'problem_statement': self.problem_statement,
            'proposed_solution': self.proposed_solution,
            'expected_contributions': self.expected_contributions,
            'key_innovations': self.key_innovations,
            'methodology': self.methodology,
            'novelty_score': self.novelty_score,
            'relevance_score': self.relevance_score,
            'feasibility_score': self.feasibility_score,
            'similar_papers': self.similar_papers or [],
            'is_novel': self.is_novel
        }


class IdeaGenerator:
    """研究想法生成器"""

    # CV 领域热点方向
    CV_TRENDS = [
        "vision_transformers",
        "diffusion_models",
        "neural_radiance_fields",
        "3d_gaussian_splatting",
        "multimodal_learning",
        "efficient_network_design",
        "self_supervised_learning",
        "foundation_models",
        "prompt_learning",
        "domain_adaptation"
    ]

    # 常见的改进角度
    IMPROVEMENT_ANGLES = [
        "architectural_modification",
        "training_strategy",
        "loss_function",
        "data_augmentation",
        "attention_mechanism",
        "normalization_technique",
        "optimization_method",
        "multi_scale_fusion",
        "knowledge_distillation",
        "ensemble_strategy"
    ]

    def __init__(self, llm_client=None):
        self.llm_client = llm_client or get_llm_client()

    def generate_ideas(
        self,
        task: str = "classification",
        dataset: str = "imagenet",
        num_ideas: int = 5,
        trending_topics: List[str] = None
    ) -> List[ResearchIdea]:
        """
        生成研究想法

        Args:
            task: 任务类型 (classification/detection/segmentation)
            dataset: 数据集
            num_ideas: 生成想法数量
            trending_topics: 指定热点方向

        Returns:
            ResearchIdea 列表
        """
        print(f"\n🎯 Generating {num_ideas} research ideas for {task} on {dataset}...")

        ideas = []
        used_combinations = set()

        for i in range(num_ideas):
            # 选择改进角度和热点方向
            angle = random.choice(self.IMPROVEMENT_ANGLES)
            trend = random.choice(trending_topics or self.CV_TRENDS)

            # 确保组合不重复
            combo = (angle, trend)
            attempts = 0
            while combo in used_combinations and attempts < 10:
                angle = random.choice(self.IMPROVEMENT_ANGLES)
                trend = random.choice(trending_topics or self.CV_TRENDS)
                combo = (angle, trend)
                attempts += 1

            used_combinations.add(combo)

            print(f"\n  Idea {i+1}/{num_ideas}: {angle} + {trend}")

            # 使用 LLM 生成详细想法
            idea = self._generate_idea_with_llm(
                task=task,
                dataset=dataset,
                improvement_angle=angle,
                trending_topic=trend,
                idea_id=f"idea_{i+1}"
            )

            if idea:
                ideas.append(idea)
                print(f"    ✓ Generated: {idea.title}")

        return ideas

    def _generate_idea_with_llm(
        self,
        task: str,
        dataset: str,
        improvement_angle: str,
        trending_topic: str,
        idea_id: str
    ) -> Optional[ResearchIdea]:
        """使用 LLM 生成具体想法"""

        system_prompt = """You are a senior computer vision researcher specializing in generating novel research ideas for top-tier conferences like CVPR.
Your ideas should be:
1. Technically sound and feasible
2. Novel and distinct from existing work
3. Potentially impactful for the field
4. Clearly articulated with specific contributions

Generate responses in strict JSON format."""

        prompt = f"""Generate a research idea for {task} on {dataset}.

Focus Areas:
- Improvement Angle: {improvement_angle}
- Trending Topic: {trending_topic}

Please provide a detailed research idea with the following fields in JSON format:
{{
    "title": "A clear, concise title for the paper",
    "abstract": "A 150-200 word abstract describing the problem, proposed solution, and expected results",
    "problem_statement": "What specific problem does this work address? Why is it important?",
    "proposed_solution": "What is your proposed approach/method?",
    "expected_contributions": ["List 3-4 specific contributions"],
    "key_innovations": ["List 2-3 key technical innovations"],
    "methodology": "Brief description of the approach and experiments"
}}

Make the idea novel and specific. Avoid generic descriptions."""

        try:
            response = self.llm_client.generate(prompt, system_prompt)

            if response.error:
                print(f"    ⚠️ LLM error: {response.error}")
                return None

            # 解析 JSON 响应
            content = response.content

            # 尝试提取 JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                return ResearchIdea(
                    id=idea_id,
                    title=data.get('title', f'Untitled Idea {idea_id}'),
                    abstract=data.get('abstract', ''),
                    problem_statement=data.get('problem_statement', ''),
                    proposed_solution=data.get('proposed_solution', ''),
                    expected_contributions=data.get('expected_contributions', []),
                    key_innovations=data.get('key_innovations', []),
                    methodology=data.get('methodology', '')
                )
            else:
                print(f"    ⚠️ Could not parse JSON from response")
                return None

        except json.JSONDecodeError as e:
            print(f"    ⚠️ JSON parsing error: {e}")
            return None
        except Exception as e:
            print(f"    ⚠️ Error generating idea: {e}")
            return None

    def evaluate_idea_quality(self, idea: ResearchIdea) -> Dict[str, float]:
        """评估想法质量"""

        system_prompt = "You are an expert reviewer for CVPR. Evaluate research ideas objectively."

        prompt = f"""Evaluate the following research idea on a scale of 1-10 for each criterion:

Title: {idea.title}
Abstract: {idea.abstract}
Key Innovations: {', '.join(idea.key_innovations)}

Criteria:
1. Novelty: How novel is this idea compared to existing work?
2. Relevance: How relevant is this to current CV challenges?
3. Feasibility: How feasible is this to implement in 3-6 months?
4. Impact: What is the potential impact if successful?

Respond in JSON format:
{{
    "novelty": 7.5,
    "relevance": 8.0,
    "feasibility": 7.0,
    "impact": 7.5
}}"""

        try:
            response = self.llm_client.generate(prompt, system_prompt)

            if response.error:
                # 使用默认分数
                return {'novelty': 6.0, 'relevance': 6.5, 'feasibility': 6.0, 'impact': 6.0}

            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                idea.novelty_score = scores.get('novelty', 6.0)
                idea.relevance_score = scores.get('relevance', 6.5)
                idea.feasibility_score = scores.get('feasibility', 6.0)
                return scores

        except Exception as e:
            print(f"⚠️ Error evaluating idea: {e}")

        return {'novelty': 6.0, 'relevance': 6.5, 'feasibility': 6.0, 'impact': 6.0}


class NoveltyChecker:
    """新颖性检查器"""

    def __init__(self, llm_client=None, search_engine: str = "openalex"):
        self.llm_client = llm_client or get_llm_client()
        self.search_engine = search_engine

    def check_novelty(self, idea: ResearchIdea) -> Tuple[bool, List[Dict]]:
        """
        检查想法的新颖性

        Returns:
            (is_novel, similar_papers)
        """
        print(f"\n🔍 Checking novelty of: {idea.title}")

        # 使用 LLM 模拟相关工作检查
        # 实际应用中应该调用 Semantic Scholar 或 OpenAlex API

        similar_papers = self._search_similar_papers(idea)

        if similar_papers:
            print(f"  Found {len(similar_papers)} potentially similar papers")

            # 评估相似度
            is_novel = self._assess_novelty_vs_papers(idea, similar_papers)

            if not is_novel:
                print("  ⚠️ Idea may not be sufficiently novel")
        else:
            print("  ✓ No highly similar papers found")
            is_novel = True

        idea.similar_papers = similar_papers
        idea.is_novel = is_novel

        return is_novel, similar_papers

    def _search_similar_papers(self, idea: ResearchIdea) -> List[Dict]:
        """搜索相关论文（模拟）"""
        # 实际实现应该调用:
        # - Semantic Scholar API
        # - OpenAlex API
        # - Google Scholar (通过 scholarly 库)

        # 这里使用 LLM 模拟搜索结果
        system_prompt = "You are a literature search assistant. Identify papers that might be related to a given research idea."

        prompt = f"""Given this research idea:

Title: {idea.title}
Abstract: {idea.abstract}
Key Innovations: {', '.join(idea.key_innovations)}

List 2-3 papers that might be most similar to this work. For each paper, provide:
- Title
- Authors (year)
- Brief description of similarity
- Estimated overlap percentage (0-100%)

Respond in JSON format:
[
    {{
        "title": "Paper Title",
        "authors": "Author et al.",
        "year": 2023,
        "similarity": "Brief description",
        "overlap_percentage": 30
    }}
]

If no highly relevant papers, return an empty list []."""

        try:
            response = self.llm_client.generate(prompt, system_prompt)

            if response.error:
                return []

            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                papers = json.loads(json_match.group())
                # 过滤掉低相似度的
                return [p for p in papers if p.get('overlap_percentage', 0) > 20]

        except Exception as e:
            print(f"⚠️ Error searching papers: {e}")

        return []

    def _assess_novelty_vs_papers(self, idea: ResearchIdea, papers: List[Dict]) -> bool:
        """评估相对于已有论文的新颖性"""

        system_prompt = "You are an expert reviewer assessing the novelty of research proposals."

        papers_desc = "\n\n".join([
            f"Paper {i+1}: {p['title']} ({p['authors']}, {p['year']})\n"
            f"Similarity: {p['similarity']}\n"
            f"Estimated overlap: {p['overlap_percentage']}%"
            for i, p in enumerate(papers)
        ])

        prompt = f"""Given this proposed research:

Title: {idea.title}
Key Innovations: {', '.join(idea.key_innovations)}

And these existing papers:
{papers_desc}

Assess whether the proposed idea is sufficiently novel to warrant a new paper.

Consider:
1. Is the core contribution distinct from existing work?
2. Does it address a different problem or use a fundamentally different approach?
3. Would this be considered incremental or a meaningful advance?

Respond with JSON:
{{
    "is_sufficiently_novel": true/false,
    "reasoning": "Brief explanation",
    "recommended_action": "Proceed/Modify/Reject"
}}"""

        try:
            response = self.llm_client.generate(prompt, system_prompt)

            if response.error:
                return True  # 默认通过

            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get('is_sufficiently_novel', True)

        except Exception as e:
            print(f"⚠️ Error assessing novelty: {e}")

        return True


class IdeaSelectionPipeline:
    """想法选择和优化管道"""

    def __init__(self, llm_client=None):
        self.generator = IdeaGenerator(llm_client)
        self.checker = NoveltyChecker(llm_client)

    def generate_and_select(
        self,
        task: str,
        dataset: str,
        num_candidates: int = 10,
        top_k: int = 3
    ) -> List[ResearchIdea]:
        """
        生成并选择最佳想法

        Returns:
            按质量排序的前 k 个想法
        """
        print("="*70)
        print("🎯 IDEA GENERATION & SELECTION PIPELINE")
        print("="*70)

        # 1. 生成候选想法
        candidates = self.generator.generate_ideas(
            task=task,
            dataset=dataset,
            num_ideas=num_candidates
        )

        if not candidates:
            print("❌ No ideas generated successfully")
            return []

        print(f"\n✓ Generated {len(candidates)} candidate ideas")

        # 2. 评估每个想法
        print("\n📊 Evaluating idea quality...")
        for idea in candidates:
            scores = self.generator.evaluate_idea_quality(idea)
            print(f"  {idea.id}: Novelty={scores['novelty']:.1f}, "
                  f"Feasibility={scores['feasibility']:.1f}")

        # 3. 检查新颖性
        print("\n🔍 Checking novelty...")
        for idea in candidates:
            is_novel, similar = self.checker.check_novelty(idea)
            status = "✓ Novel" if is_novel else "⚠️ Similar work exists"
            print(f"  {idea.id}: {status}")

        # 4. 筛选和排序
        # 过滤掉不够新颖的
        novel_candidates = [i for i in candidates if i.is_novel]

        if not novel_candidates:
            print("⚠️ No novel ideas found, using best candidates anyway")
            novel_candidates = candidates

        # 按综合分数排序
        def score_idea(idea):
            return (
                idea.novelty_score * 0.4 +
                idea.relevance_score * 0.2 +
                idea.feasibility_score * 0.3 +
                (10 - len(idea.similar_papers or []) * 2) * 0.1  # 相关论文越少越好
            )

        ranked = sorted(novel_candidates, key=score_idea, reverse=True)

        # 5. 选择前 k 个
        selected = ranked[:top_k]

        print(f"\n🏆 Selected Top {len(selected)} Ideas:")
        for i, idea in enumerate(selected, 1):
            print(f"\n  {i}. {idea.title}")
            print(f"     Novelty: {idea.novelty_score:.1f}/10 | "
                  f"Feasible: {idea.feasibility_score:.1f}/10")
            print(f"     Key Innovation: {idea.key_innovations[0] if idea.key_innovations else 'N/A'}")

        return selected


if __name__ == "__main__":
    # 测试
    print("Testing Idea Generation Pipeline...\n")

    pipeline = IdeaSelectionPipeline()

    ideas = pipeline.generate_and_select(
        task="classification",
        dataset="cifar10",
        num_candidates=3,
        top_k=2
    )

    print("\n\nFinal Selected Idea:")
    if ideas:
        print(json.dumps(ideas[0].to_dict(), indent=2))
