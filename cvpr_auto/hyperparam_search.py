"""
超参数自动搜索模块
支持 Optuna 和 Ray Tune
"""

import optuna
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
from typing import Dict, Callable, Optional
import json
import os
from pathlib import Path


class HyperParamSearcher:
    """超参数搜索器"""

    def __init__(self, search_space: Dict, n_trials: int = 50):
        self.search_space = search_space
        self.n_trials = n_trials
        self.best_params = None
        self.study = None

    def suggest_params(self, trial: optuna.Trial) -> Dict:
        """从搜索空间建议参数"""
        params = {}

        for name, config in self.search_space.items():
            param_type = config.get('type', 'float')

            if param_type == 'log':
                params[name] = trial.suggest_float(
                    name, config['low'], config['high'], log=True
                )
            elif param_type == 'choice':
                params[name] = trial.suggest_categorical(
                    name, config['options']
                )
            elif param_type == 'int':
                params[name] = trial.suggest_int(
                    name, config['low'], config['high']
                )
            elif param_type == 'float':
                params[name] = trial.suggest_float(
                    name, config['low'], config['high']
                )

        return params

    def search(self, objective_fn: Callable[[Dict], float]) -> Dict:
        """
        执行超参数搜索

        Args:
            objective_fn: 接受参数字典，返回验证指标（越高越好）

        Returns:
            最佳参数
        """
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )

        def wrapped_objective(trial):
            params = self.suggest_params(trial)
            score = objective_fn(params)
            return score

        self.study.optimize(wrapped_objective, n_trials=self.n_trials)

        self.best_params = self.study.best_params

        print(f"\n{'='*60}")
        print(f"Best score: {self.study.best_value:.4f}")
        print(f"Best params:")
        for k, v in self.best_params.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}\n")

        return self.best_params

    def save_results(self, output_dir: str):
        """保存搜索结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存最佳参数
        with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
            json.dump(self.best_params, f, indent=2)

        # 保存所有 trial
        trials_df = self.study.trials_dataframe()
        trials_df.to_csv(os.path.join(output_dir, 'trials.csv'), index=False)

        # 保存优化历史图
        try:
            fig = optuna.visualization.plot_optimization_history(self.study)
            fig.write_html(os.path.join(output_dir, 'optimization_history.html'))

            fig = optuna.visualization.plot_param_importances(self.study)
            fig.write_html(os.path.join(output_dir, 'param_importances.html'))
        except:
            pass


class AutoAblation:
    """自动消融实验设计器"""

    def __init__(self, base_config: Dict):
        self.base_config = base_config
        self.ablation_results = {}

    def generate_ablation_configs(self, component_names: list) -> list:
        """
        生成消融实验配置

        例如: ['attention', 'normalization', 'data_augmentation']
        生成:
            - 完整模型
            - 去掉 attention
            - 去掉 normalization
            - 去掉 data_augmentation
        """
        configs = []

        # 1. 完整模型
        configs.append({
            'name': 'full_model',
            'modifications': {},
            'description': 'Complete model with all components'
        })

        # 2. 去掉每个组件
        for component in component_names:
            configs.append({
                'name': f'w/o_{component}',
                'modifications': {component: False},
                'description': f'Remove {component}'
            })

        # 3. 只保留单个组件（可选）
        # for component in component_names:
        #     configs.append({
        #         'name': f'only_{component}',
        #         'modifications': {c: c == component for c in component_names},
        #         'description': f'Only use {component}'
        #     })

        return configs

    def run_ablation(self, component_names: list, train_fn: Callable) -> Dict:
        """
        运行完整消融实验

        Args:
            component_names: 组件名称列表
            train_fn: 训练函数，接受配置返回验证指标

        Returns:
            消融结果字典
        """
        configs = self.generate_ablation_configs(component_names)

        for config in configs:
            print(f"\nRunning ablation: {config['name']}")
            print(f"Description: {config['description']}")
            print('-' * 40)

            result = train_fn(config['modifications'])

            self.ablation_results[config['name']] = {
                'description': config['description'],
                'metrics': result
            }

        return self.ablation_results

    def analyze_importance(self) -> Dict:
        """分析每个组件的重要性"""
        if 'full_model' not in self.ablation_results:
            return {}

        full_score = self.ablation_results['full_model']['metrics']

        importance = {}
        for name, result in self.ablation_results.items():
            if name == 'full_model':
                continue

            component = name.replace('w/o_', '')
            drop = full_score - result['metrics']
            importance[component] = {
                'absolute_drop': drop,
                'relative_drop': drop / full_score * 100 if full_score != 0 else 0
            }

        return importance

    def save_results(self, output_dir: str):
        """保存消融结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存原始结果
        with open(os.path.join(output_dir, 'ablation_results.json'), 'w') as f:
            json.dump(self.ablation_results, f, indent=2)

        # 保存重要性分析
        importance = self.analyze_importance()
        with open(os.path.join(output_dir, 'component_importance.json'), 'w') as f:
            json.dump(importance, f, indent=2)

        # 生成表格（用于论文）
        self._generate_latex_table(output_dir)

    def _generate_latex_table(self, output_dir: str):
        """生成 LaTeX 表格"""
        lines = [
            '\\begin{table}[t]',
            '\\centering',
            '\\caption{Ablation study on key components}',
            '\\label{tab:ablation}',
            '\\begin{tabular}{lc}',
            '\\toprule',
            'Configuration & Accuracy (\\%) \\\\\
',
            '\\midrule'
        ]

        for name, result in self.ablation_results.items():
            acc = result['metrics']
            display_name = name.replace('_', ' ').replace('w/o', 'w/o').title()
            lines.append(f'{display_name} & {acc:.2f} \\\\\\n')

        lines.extend([
            '\\bottomrule',
            '\\end{tabular}',
            '\\end{table}'
        ])

        with open(os.path.join(output_dir, 'ablation_table.tex'), 'w') as f:
            f.writelines(lines)


if __name__ == '__main__':
    # 示例搜索空间
    search_space = {
        'learning_rate': {'type': 'log', 'low': 1e-5, 'high': 1e-2},
        'batch_size': {'type': 'choice', 'options': [64, 128, 256]},
        'optimizer': {'type': 'choice', 'options': ['adamw', 'sgd']},
    }

    # 示例使用
    searcher = HyperParamSearcher(search_space, n_trials=10)

    def dummy_objective(params):
        print(f"Testing params: {params}")
        # 这里应该是实际的训练代码
        return params['learning_rate'] * 1000  # dummy

    best = searcher.search(dummy_objective)
    searcher.save_results('outputs/hyperparam_search')
