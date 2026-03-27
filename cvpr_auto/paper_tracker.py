"""
论文跟踪与想法记录模块
自动抓取、解析、分析最新顶会论文
支持边读边记录研究想法
"""

import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import time

from cvpr_auto.llm_client import get_llm_client, LLMResponse
from cvpr_auto.config import config


@dataclass
class Paper:
    """论文数据结构"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    url: str
    pdf_url: Optional[str]
    published_date: str
    venue: str  # CVPR/ICCV/ECCV/NeurIPS/ICLR
    year: int
    keywords: List[str] = None
    full_text: str = None  # 缓存完整文本

    # 分析结果
    problem_statement: str = None
    method_summary: str = None
    key_contributions: List[str] = None
    limitations: List[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ResearchGap:
    """研究空白"""
    description: str
    supporting_papers: List[str]  # 相关论文 ID
    potential_approach: str
    impact_score: float  # 1-10
    feasibility_score: float  # 1-10


@dataclass
class IdeaNote:
    """边读边记录的想法"""
    id: str
    timestamp: str
    source_paper_id: str  # 从哪篇论文获得的灵感
    source_paper_title: str
    trigger_sentence: str  # 哪句话触发了想法
    idea_description: str
    potential_improvement: str
    related_papers: List[str]
    priority: str  # high/medium/low


class PaperFetcher:
    """论文获取器 - 从多个来源抓取论文"""

    SOURCES = {
        'cvpr': 'https://openaccess.thecvf.com/',
        'arxiv': 'https://arxiv.org/',
        'semantic_scholar': 'https://api.semanticscholar.org/'
    }

    def __init__(self, cache_dir: str = './.paper_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.llm_client = get_llm_client()

    def fetch_recent_papers(
        self,
        venues: List[str] = None,
        days: int = 30,
        keywords: List[str] = None,
        max_papers: int = 100
    ) -> List[Paper]:
        """
        获取近期论文

        Args:
            venues: 会议列表 ['CVPR', 'ICCV', 'ECCV', 'NeurIPS', 'ICLR']
            days: 最近多少天
            keywords: 关键词过滤
            max_papers: 最大数量
        """
        print(f"📚 Fetching recent papers from {venues or 'all venues'}...")

        papers = []

        # 方法 1: 从 arXiv API 获取
        arxiv_papers = self._fetch_from_arxiv(venues, days, keywords, max_papers//2)
        papers.extend(arxiv_papers)

        # 方法 2: 从 OpenAccess (CVF) 获取
        if not venues or any(v in ['CVPR', 'ICCV', 'ECCV'] for v in venues):
            cvf_papers = self._fetch_from_cvf(days, max_papers//3)
            papers.extend(cvf_papers)

        # 方法 3: 从 Semantic Scholar 获取
        ss_papers = self._fetch_from_semantic_scholar(venues, days, max_papers//3)
        papers.extend(ss_papers)

        # 去重
        papers = self._deduplicate_papers(papers)

        # 过滤关键词
        if keywords:
            papers = self._filter_by_keywords(papers, keywords)

        print(f"✓ Fetched {len(papers)} papers")
        return papers[:max_papers]

    def _fetch_from_arxiv(
        self,
        venues: List[str],
        days: int,
        keywords: List[str],
        max_results: int
    ) -> List[Paper]:
        """从 arXiv 获取论文"""
        try:
            import arxiv

            # 构建搜索查询
            query_parts = ['cat:cs.CV']
            if keywords:
                query_parts.extend(keywords)

            query = ' AND '.join(query_parts)

            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )

            papers = []
            for result in search.results():
                paper = Paper(
                    id=f"arxiv_{result.entry_id.split('/')[-1]}",
                    title=result.title,
                    authors=[str(a) for a in result.authors],
                    abstract=result.summary,
                    url=result.entry_id,
                    pdf_url=result.pdf_url,
                    published_date=result.published.isoformat(),
                    venue='arXiv',
                    year=result.published.year
                )
                papers.append(paper)

            return papers

        except ImportError:
            print("⚠️ arxiv package not installed. Install: pip install arxiv")
            return []
        except Exception as e:
            print(f"⚠️ Error fetching from arXiv: {e}")
            return []

    def _fetch_from_cvf(self, days: int, max_results: int) -> List[Paper]:
        """从 CVF OpenAccess 获取 CVPR/ICCV/ECCV 论文"""
        # 简化实现：实际应该爬取 HTML
        # 这里返回模拟数据或从缓存加载

        cache_file = self.cache_dir / 'cvf_papers.json'
        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
                return [Paper(**p) for p in data[:max_results]]

        return []

    def _fetch_from_semantic_scholar(
        self,
        venues: List[str],
        days: int,
        max_results: int
    ) -> List[Paper]:
        """从 Semantic Scholar API 获取"""
        try:
            import requests

            # 使用 S2 API
            url = "https://api.semanticscholar.org/graph/v1/paper/search"

            query = 'computer vision'
            if venues:
                query += ' ' + ' '.join(venues)

            params = {
                'query': query,
                'fields': 'title,authors,year,abstract,url,openAccessPdf',
                'limit': max_results,
                'publicationDateOrYear': f'{datetime.now().year - 1}:',
            }

            response = requests.get(url, params=params)
            data = response.json()

            papers = []
            for item in data.get('data', []):
                paper = Paper(
                    id=f"s2_{item.get('paperId', '')}",
                    title=item.get('title', ''),
                    authors=[a.get('name', '') for a in item.get('authors', [])],
                    abstract=item.get('abstract', ''),
                    url=item.get('url', ''),
                    pdf_url=item.get('openAccessPdf', {}).get('url'),
                    published_date=item.get('publicationDate', ''),
                    venue=item.get('venue', 'Unknown'),
                    year=item.get('year', 2024)
                )
                papers.append(paper)

            return papers

        except Exception as e:
            print(f"⚠️ Error fetching from Semantic Scholar: {e}")
            return []

    def _deduplicate_papers(self, papers: List[Paper]) -> List[Paper]:
        """去重"""
        seen_titles = set()
        unique_papers = []

        for paper in papers:
            # 标准化标题用于去重
            normalized_title = paper.title.lower().strip()
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_papers.append(paper)

        return unique_papers

    def _filter_by_keywords(self, papers: List[Paper], keywords: List[str]) -> List[Paper]:
        """关键词过滤"""
        filtered = []
        keywords_lower = [k.lower() for k in keywords]

        for paper in papers:
            text = f"{paper.title} {paper.abstract}".lower()
            if any(k in text for k in keywords_lower):
                filtered.append(paper)

        return filtered


class PaperAnalyzer:
    """论文分析器 - 使用 LLM 深入分析论文"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client or get_llm_client()

    def analyze_paper(self, paper: Paper) -> Paper:
        """分析单篇论文"""
        print(f"  Analyzing: {paper.title[:60]}...")

        system_prompt = """You are an expert computer vision researcher analyzing papers for research gap identification.
Provide concise, structured analysis focusing on:
1. Problem statement
2. Method summary
3. Key contributions
4. Limitations or unresolved issues"""

        prompt = f"""Analyze the following research paper:

Title: {paper.title}
Abstract: {paper.abstract}

Provide a structured analysis in JSON format:
{{
    "problem_statement": "What problem does this paper address? (1-2 sentences)",
    "method_summary": "Brief summary of the proposed method",
    "key_contributions": ["List 2-3 key contributions"],
    "limitations": ["List 1-2 limitations or unresolved issues"]
}}

Focus on identifying what this paper does NOT solve."""

        try:
            response = self.llm_client.generate(prompt, system_prompt)

            if not response.error:
                # 提取 JSON
                import re
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())

                    paper.problem_statement = analysis.get('problem_statement', '')
                    paper.method_summary = analysis.get('method_summary', '')
                    paper.key_contributions = analysis.get('key_contributions', [])
                    paper.limitations = analysis.get('limitations', [])

        except Exception as e:
            print(f"    ⚠️ Analysis failed: {e}")

        return paper

    def analyze_papers_batch(self, papers: List[Paper]) -> List[Paper]:
        """批量分析论文"""
        print(f"\n🔍 Analyzing {len(papers)} papers...")

        analyzed = []
        for i, paper in enumerate(papers, 1):
            print(f"  [{i}/{len(papers)}]", end=' ')
            analyzed_paper = self.analyze_paper(paper)
            analyzed.append(analyzed_paper)
            time.sleep(0.5)  # 避免 API 限制

        return analyzed


class GapAnalyzer:
    """研究空白分析器 - 识别研究趋势和空白"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client or get_llm_client()

    def identify_gaps(self, papers: List[Paper]) -> List[ResearchGap]:
        """基于分析后的论文识别研究空白"""
        print("\n🔎 Identifying research gaps...")

        # 准备论文摘要
        papers_text = "\n\n".join([
            f"Paper {i+1}: {p.title}\n"
            f"Problem: {p.problem_statement or 'N/A'}\n"
            f"Limitations: {', '.join(p.limitations or [])}"
            for i, p in enumerate(papers[:20])  # 限制数量
        ])

        system_prompt = """You are a research director identifying promising research directions.
Look for patterns, unresolved issues, and opportunities for improvement."""

        prompt = f"""Based on the following recent papers, identify 3-5 research gaps or opportunities:

{papers_text}

For each gap, provide:
1. Clear description of what's missing
2. Which papers support this gap (by number)
3. A potential approach to address it
4. Impact score (1-10)
5. Feasibility score (1-10)

Respond in JSON format:
[
    {{
        "description": "Clear description of the research gap",
        "supporting_papers": [1, 3, 5],
        "potential_approach": "How to address this gap",
        "impact_score": 8,
        "feasibility_score": 7
    }}
]"""

        try:
            response = self.llm_client.generate(prompt, system_prompt)

            if response.error:
                return []

            # 提取 JSON
            import re
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                gaps_data = json.loads(json_match.group())

                gaps = []
                for gap_data in gaps_data:
                    # 转换 paper 编号为 ID
                    paper_indices = gap_data.get('supporting_papers', [])
                    supporting_ids = [papers[i-1].id for i in paper_indices if 0 < i <= len(papers)]

                    gap = ResearchGap(
                        description=gap_data.get('description', ''),
                        supporting_papers=supporting_ids,
                        potential_approach=gap_data.get('potential_approach', ''),
                        impact_score=gap_data.get('impact_score', 5),
                        feasibility_score=gap_data.get('feasibility_score', 5)
                    )
                    gaps.append(gap)

                print(f"✓ Identified {len(gaps)} research gaps")
                return gaps

        except Exception as e:
            print(f"⚠️ Gap analysis failed: {e}")

        return []


class IdeaRecorder:
    """想法记录器 - 边读边记录灵感"""

    def __init__(self, notes_file: str = './research_notes.json'):
        self.notes_file = Path(notes_file)
        self.notes: List[IdeaNote] = []
        self.load_notes()

    def load_notes(self):
        """加载已有笔记"""
        if self.notes_file.exists():
            with open(self.notes_file) as f:
                data = json.load(f)
                self.notes = [IdeaNote(**n) for n in data]

    def save_notes(self):
        """保存笔记"""
        with open(self.notes_file, 'w') as f:
            json.dump([asdict(n) for n in self.notes], f, indent=2)

    def record_idea(
        self,
        paper: Paper,
        trigger_text: str,
        idea_description: str,
        improvement: str,
        priority: str = 'medium'
    ) -> IdeaNote:
        """记录一个想法"""
        note = IdeaNote(
            id=f"note_{len(self.notes) + 1}_{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            source_paper_id=paper.id,
            source_paper_title=paper.title,
            trigger_sentence=trigger_text[:200],  # 限制长度
            idea_description=idea_description,
            potential_improvement=improvement,
            related_papers=[paper.id],
            priority=priority
        )

        self.notes.append(note)
        self.save_notes()

        print(f"✓ Recorded idea: {idea_description[:60]}...")
        return note

    def generate_ideas_from_notes(self, llm_client=None) -> List[Dict]:
        """基于记录的笔记生成结构化研究想法"""
        if not self.notes:
            return []

        print(f"\n💡 Generating research ideas from {len(self.notes)} notes...")

        # 准备笔记文本
        notes_text = "\n\n".join([
            f"Note {i+1} (from {n.source_paper_title}):\n"
            f"Trigger: {n.trigger_sentence}\n"
            f"Idea: {n.idea_description}\n"
            f"Improvement: {n.potential_improvement}"
            for i, n in enumerate(self.notes)
        ])

        llm = llm_client or get_llm_client()

        system_prompt = """You are a research strategist synthesizing scattered ideas into concrete research proposals.
Combine related thoughts and identify the most promising directions."""

        prompt = f"""Based on the following research notes collected while reading papers, synthesize 3-5 concrete research ideas:

{notes_text}

For each idea, provide:
1. A clear, specific title
2. Problem statement
3. Proposed solution
4. Expected contributions
5. Feasibility assessment

Respond in JSON format:
[
    {{
        "title": "Specific, descriptive title",
        "problem": "Clear problem statement",
        "solution": "Proposed approach",
        "contributions": ["List 2-3 contributions"],
        "feasibility": "high/medium/low"
    }}
]"""

        try:
            response = llm.generate(prompt, system_prompt)

            if response.error:
                return []

            import re
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                ideas = json.loads(json_match.group())
                print(f"✓ Generated {len(ideas)} synthesized ideas from notes")
                return ideas

        except Exception as e:
            print(f"⚠️ Idea generation from notes failed: {e}")

        return []


class PaperDrivenIdeaPipeline:
    """文献驱动的想法生成管道"""

    def __init__(self):
        self.fetcher = PaperFetcher()
        self.analyzer = PaperAnalyzer()
        self.gap_analyzer = GapAnalyzer()
        self.idea_recorder = IdeaRecorder()
        self.llm_client = get_llm_client()

    def run_full_pipeline(
        self,
        venues: List[str] = None,
        keywords: List[str] = None,
        days: int = 30,
        max_papers: int = 50
    ) -> Tuple[List[ResearchGap], List[Dict]]:
        """
        运行完整管道

        Returns:
            (research_gaps, synthesized_ideas)
        """
        print("=" * 70)
        print("📚 Paper-Driven Research Idea Generation Pipeline")
        print("=" * 70)

        # Step 1: 获取论文
        papers = self.fetcher.fetch_recent_papers(
            venues=venues,
            days=days,
            keywords=keywords,
            max_papers=max_papers
        )

        if not papers:
            print("❌ No papers fetched")
            return [], []

        # Step 2: 分析论文
        analyzed_papers = self.analyzer.analyze_papers_batch(papers)

        # Step 3: 识别研究空白
        gaps = self.gap_analyzer.identify_gaps(analyzed_papers)

        # Step 4: 基于空白记录想法（模拟边读边想）
        print("\n📝 Recording ideas from gaps...")
        for gap in gaps[:3]:  # 前 3 个空白
            # 找到支持的论文
            supporting_paper = next(
                (p for p in analyzed_papers if p.id in gap.supporting_papers),
                analyzed_papers[0]
            )

            self.idea_recorder.record_idea(
                paper=supporting_paper,
                trigger_text=gap.description,
                idea_description=gap.description,
                improvement=gap.potential_approach,
                priority='high' if gap.impact_score >= 8 else 'medium'
            )

        # Step 5: 从笔记生成结构化想法
        ideas = self.idea_recorder.generate_ideas_from_notes(self.llm_client)

        # 保存完整结果
        self._save_results(analyzed_papers, gaps, ideas)

        return gaps, ideas

    def _save_results(self, papers: List[Paper], gaps: List[ResearchGap], ideas: List[Dict]):
        """保存结果"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'papers': [p.to_dict() for p in papers],
            'gaps': [asdict(g) for g in gaps],
            'ideas': ideas
        }

        output_file = Path('./cvpr_outputs/paper_driven_ideas.json')
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n💾 Results saved to {output_file}")


if __name__ == "__main__":
    # 测试
    pipeline = PaperDrivenIdeaPipeline()

    gaps, ideas = pipeline.run_full_pipeline(
        venues=['CVPR', 'ICCV'],
        keywords=['vision transformer', 'efficient'],
        days=90,
        max_papers=20
    )

    print("\n" + "=" * 70)
    print("RESEARCH GAPS IDENTIFIED:")
    print("=" * 70)
    for i, gap in enumerate(gaps, 1):
        print(f"\n{i}. {gap.description}")
        print(f"   Impact: {gap.impact_score}/10 | Feasibility: {gap.feasibility_score}/10")
        print(f"   Approach: {gap.potential_approach[:100]}...")
