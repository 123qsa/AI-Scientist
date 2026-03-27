"""
CVPR 专业论文生成器
支持高质量图表、专业写作、LaTeX 输出
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import os
from pathlib import Path


class CVPRPaperComposer:
    """CVPR 论文撰写器"""

    def __init__(self, output_dir: str = "./cvpr_paper"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # CVPR 论文结构
        self.sections = {
            'abstract': '',
            'introduction': '',
            'related_work': '',
            'method': '',
            'experiments': '',
            'conclusion': '',
        }

        self.figures = []
        self.tables = []

    def generate_figures(self, experiment_results: Dict) -> List[Path]:
        """生成专业论文图表"""
        figure_paths = []

        # Figure 1: 训练曲线
        if 'train_history' in experiment_results:
            fig_path = self._plot_training_curves(
                experiment_results['train_history']
            )
            figure_paths.append(fig_path)

        # Figure 2: 与 SOTA 对比
        if 'sota_comparison' in experiment_results:
            fig_path = self._plot_sota_comparison(
                experiment_results['sota_comparison']
            )
            figure_paths.append(fig_path)

        # Figure 3: 消融实验可视化
        if 'ablation' in experiment_results:
            fig_path = self._plot_ablation(
                experiment_results['ablation']
            )
            figure_paths.append(fig_path)

        # Figure 4: 超参搜索可视化
        if 'hyperparam_search' in experiment_results:
            fig_path = self._plot_hyperparam_importance(
                experiment_results['hyperparam_search']
            )
            figure_paths.append(fig_path)

        self.figures = figure_paths
        return figure_paths

    def _plot_training_curves(self, history: Dict) -> Path:
        """绘制训练曲线 (Figure 1)"""
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))

        # Loss 曲线
        ax = axes[0]
        epochs = range(len(history.get('train_loss', [])))
        ax.plot(epochs, history['train_loss'], label='Train', linewidth=1.5)
        ax.plot(epochs, history.get('val_loss', []), label='Val', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy 曲线
        ax = axes[1]
        ax.plot(epochs, history.get('train_acc', []), label='Train', linewidth=1.5)
        ax.plot(epochs, history.get('val_acc', []), label='Val', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Top-1 Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        fig_path = self.output_dir / 'fig1_training_curves.pdf'
        plt.savefig(fig_path, format='pdf', bbox_inches='tight')
        plt.close()

        return fig_path

    def _plot_sota_comparison(self, sota_data: Dict) -> Path:
        """绘制 SOTA 对比 (Figure 2)"""
        methods = list(sota_data.keys())
        scores = list(sota_data.values())

        # 排序
        sorted_data = sorted(zip(methods, scores), key=lambda x: x[1])
        methods, scores = zip(*sorted_data)

        fig, ax = plt.subplots(figsize=(6, 4))

        colors = ['#1f77b4'] * (len(methods) - 1) + ['#d62728']  # 最后一个是我们的方法
        bars = ax.barh(range(len(methods)), scores, color=colors)

        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, scores)):
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{score:.2f}', ha='left', va='center', fontsize=9)

        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_xlabel('Top-1 Accuracy (%)')
        ax.set_title('Comparison with State-of-the-Art on ImageNet-1K')
        ax.grid(True, axis='x', alpha=0.3)

        # 添加年份/会议标记
        ax.axvline(x=max(scores[:-1]) if len(scores) > 1 else 0,
                  color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()

        fig_path = self.output_dir / 'fig2_sota_comparison.pdf'
        plt.savefig(fig_path, format='pdf', bbox_inches='tight')
        plt.close()

        return fig_path

    def _plot_ablation(self, ablation_data: Dict) -> Path:
        """绘制消融实验 (Figure 3)"""
        configs = list(ablation_data.keys())
        scores = [ablation_data[k]['metrics'] for k in configs]

        fig, ax = plt.subplots(figsize=(6, 3))

        # 突出完整模型
        colors = ['#d62728' if c == 'full_model' else '#1f77b4' for c in configs]

        bars = ax.bar(range(len(configs)), scores, color=colors, width=0.6)

        # 添加数值标签
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                   f'{score:.2f}', ha='center', va='bottom', fontsize=9)

        # 格式化标签
        labels = [c.replace('w/o_', 'w/o ').replace('_', ' ').title() for c in configs]
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(labels, rotation=45, ha='right')

        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Ablation Study on Key Components')
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()

        fig_path = self.output_dir / 'fig3_ablation.pdf'
        plt.savefig(fig_path, format='pdf', bbox_inches='tight')
        plt.close()

        return fig_path

    def _plot_hyperparam_importance(self, search_results: Dict) -> Path:
        """绘制超参重要性 (Figure 4)"""
        if 'param_importance' not in search_results:
            return None

        importance = search_results['param_importance']
        params = list(importance.keys())
        values = list(importance.values())

        # 排序
        sorted_data = sorted(zip(params, values), key=lambda x: x[1], reverse=True)
        params, values = zip(*sorted_data)

        fig, ax = plt.subplots(figsize=(5, 3))

        ax.barh(range(len(params)), values, color='#2ca02c')
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(params)
        ax.set_xlabel('Importance')
        ax.set_title('Hyperparameter Importance (Optuna)')
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()

        fig_path = self.output_dir / 'fig4_hyperparam_importance.pdf'
        plt.savefig(fig_path, format='pdf', bbox_inches='tight')
        plt.close()

        return fig_path

    def generate_latex_paper(self, paper_content: Dict, bib_entries: List[Dict]) -> Path:
        """生成完整 LaTeX 论文"""

        latex_lines = [
            '\\documentclass[10pt,twocolumn,letterpaper]{article}',
            '',
            '% CVPR 2025 packages',
            '\\usepackage[review]{cvpr}',
            '\\usepackage{times}',
            '\\usepackage{epsfig}',
            '\\usepackage{graphicx}',
            '\\usepackage{amsmath}',
            '\\usepackage{amssymb}',
            '\\usepackage{booktabs}',
            '\\usepackage{multirow}',
            '\\usepackage{color}',
            '\\usepackage{cite}',
            '\\usepackage{algorithm}',
            '\\usepackage{algorithmic}',
            '',
            '% Include other packages here, before hyperref',
            '\\usepackage[pagebackref=true,breaklinks=true,letterpaper=true,colorlinks,bookmarks=false]{hyperref}',
            '',
            '\\cvprfinalcopy',
            '',
            '\\def\\cvprPaperID{****}',
            '\\def\\httilde{\mbox{\\tt\\char126}}',
            '',
            '\\title{' + paper_content.get('title', 'Title TBD') + '}',
            '',
            '\\author{\\textit{Anonymous CVPR submission}}',
            '',
            '\\begin{document}',
            '',
            '\\maketitle',
            '',
            '% Abstract',
            '\\begin{abstract}',
            paper_content.get('abstract', 'Abstract TBD'),
            '\\end{abstract}',
            '',
            '% Introduction',
            '\\section{Introduction}',
            paper_content.get('introduction', 'Introduction TBD'),
            '',
            '% Related Work',
            '\\section{Related Work}',
            paper_content.get('related_work', 'Related work TBD'),
            '',
            '% Method',
            '\\section{Method}',
            '\\label{sec:method}',
            paper_content.get('method', 'Method TBD'),
            '',
            '% Experiments',
            '\\section{Experiments}',
            '\\label{sec:experiments}',
            paper_content.get('experiments', 'Experiments TBD'),
            '',
            '% Figures',
            '\\begin{figure}[t]',
            '\\centering',
            '\\includegraphics[width=\\linewidth]{fig1_training_curves.pdf}',
            '\\caption{Training curves on ImageNet-1K.}',
            '\\label{fig:training}',
            '\\end{figure}',
            '',
            '\\begin{figure}[t]',
            '\\centering',
            '\\includegraphics[width=\\linewidth]{fig2_sota_comparison.pdf}',
            '\\caption{Comparison with state-of-the-art methods on ImageNet-1K.}',
            '\\label{fig:sota}',
            '\\end{figure}',
            '',
            '\\begin{figure}[t]',
            '\\centering',
            '\\includegraphics[width=\\linewidth]{fig3_ablation.pdf}',
            '\\caption{Ablation study results.}',
            '\\label{fig:ablation}',
            '\\end{figure}',
            '',
            '% Conclusion',
            '\\section{Conclusion}',
            paper_content.get('conclusion', 'Conclusion TBD'),
            '',
            '% Acknowledgments (remove for submission)',
            '%\\section*{Acknowledgments}',
            '',
            '% References',
            '\\bibliographystyle{ieee_fullname}',
            '\\bibliography{references}',
            '',
            '\\end{document}',
        ]

        tex_path = self.output_dir / 'paper.tex'
        with open(tex_path, 'w') as f:
            f.write('\n'.join(latex_lines))

        # 生成参考文献文件
        self._generate_bibliography(bib_entries)

        return tex_path

    def _generate_bibliography(self, bib_entries: List[Dict]):
        """生成 BibTeX 文件"""
        bib_path = self.output_dir / 'references.bib'

        with open(bib_path, 'w') as f:
            for entry in bib_entries:
                f.write(f"@{entry['type']}{{{entry['key']},\n")
                for k, v in entry.items():
                    if k not in ['type', 'key']:
                        f.write(f"  {k} = {{{v}}},\n")
                f.write('}\n\n')

    def compile_paper(self, tex_path: Path) -> Path:
        """编译 LaTeX 生成 PDF"""
        import subprocess

        # 需要多次编译
        for _ in range(3):
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', str(tex_path)],
                cwd=self.output_dir,
                capture_output=True
            )

        # 编译 bibliography
        subprocess.run(
            ['bibtex', str(tex_path.stem)],
            cwd=self.output_dir,
            capture_output=True
        )

        # 最终编译
        for _ in range(2):
            subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', str(tex_path)],
                cwd=self.output_dir,
                capture_output=True
            )

        pdf_path = self.output_dir / f'{tex_path.stem}.pdf'
        return pdf_path if pdf_path.exists() else None


if __name__ == '__main__':
    # 示例使用
    composer = CVPRPaperComposer()

    # 示例实验结果
    results = {
        'train_history': {
            'train_loss': [2.3, 1.8, 1.5, 1.2, 1.0],
            'val_loss': [2.4, 1.9, 1.6, 1.4, 1.3],
            'train_acc': [20, 45, 60, 72, 78],
            'val_acc': [18, 42, 58, 68, 75],
        },
        'sota_comparison': {
            'ResNet-50': 76.1,
            'ResNet-101': 77.4,
            'EfficientNet-B0': 77.3,
            'Our Method': 79.5,
        },
        'ablation': {
            'full_model': {'metrics': 79.5},
            'w/o_attention': {'metrics': 77.2},
            'w/o_normalization': {'metrics': 76.8},
            'w/o_augmentation': {'metrics': 78.1},
        }
    }

    # 生成图表
    composer.generate_figures(results)

    # 生成论文
    paper_content = {
        'title': 'Novel Architecture for Image Classification',
        'abstract': 'We propose a new method that achieves state-of-the-art results...',
        'introduction': 'Deep learning has achieved remarkable success...',
        'related_work': 'Recent advances in computer vision...',
        'method': 'Our method consists of three key components...',
        'experiments': 'We evaluate our method on ImageNet-1K...',
        'conclusion': 'In this paper, we presented a novel approach...',
    }

    bib_entries = [
        {
            'type': 'inproceedings',
            'key': 'he2016deep',
            'title': 'Deep Residual Learning for Image Recognition',
            'author': 'He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian',
            'booktitle': 'CVPR',
            'year': '2016'
        }
    ]

    tex_path = composer.generate_latex_paper(paper_content, bib_entries)
    print(f"Generated: {tex_path}")
