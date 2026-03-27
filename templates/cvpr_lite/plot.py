"""
Visualization script for CVPR-Lite experiments.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history_path='run_0/training_history.json', output_dir='run_0'):
    """Plot training curves."""
    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history['train_acc']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=1.5)
    axes[0].plot(epochs, history['test_loss'], label='Test Loss', linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], label='Train Acc', linewidth=1.5)
    axes[1].plot(epochs, history['test_acc'], label='Test Acc', linewidth=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Test Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {output_dir}/")


def plot_comparison(baseline_path, improved_path, output_dir='run_0'):
    """Plot comparison between baseline and improved method."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(improved_path) as f:
        improved = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5))

    metrics = ['Final Test Acc', 'Best Test Acc']
    baseline_vals = [
        baseline['final_test_acc']['means'],
        baseline['best_test_acc']['means']
    ]
    improved_vals = [
        improved['final_test_acc']['means'],
        improved['best_test_acc']['means']
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='#1f77b4')
    bars2 = ax.bar(x + width/2, improved_vals, width, label='Proposed', color='#ff7f0e')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        if len(sys.argv) >= 5:
            plot_comparison(sys.argv[2], sys.argv[3], sys.argv[4])
        else:
            print("Usage: python plot.py --compare baseline.json improved.json output_dir")
    else:
        plot_training_curves()
