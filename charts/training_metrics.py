"""
training_metrics.py
-------------------
Training-related charts for the report.
Covers: Chapter 4 - Model Training (Baseline to Improved)

Run: python charts/training_metrics.py
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#cccccc',
    'axes.labelcolor': '#333333',
    'text.color': '#333333',
    'xtick.color': '#555555',
    'ytick.color': '#555555',
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.grid': True,
    'grid.color': '#eeeeee',
    'grid.alpha': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUT_DIR = 'charts/output'
os.makedirs(OUT_DIR, exist_ok=True)

try:
    with open('baseline_metrics.json', 'r') as f:
        baseline = json.load(f)
    epochs = list(range(1, 6))
    train_loss = [baseline['train_loss'][str(i)] for i in range(5)]
    train_acc = [baseline['train_acc'][str(i)] for i in range(5)]
    val_f1 = [baseline['val_macro_f1'][str(i)] for i in range(5)]
    val_acc = [baseline['val_acc'][str(i)] for i in range(5)]
except FileNotFoundError:
    epochs = [1, 2, 3, 4, 5]
    train_loss = [0.4607, 0.2830, 0.2179, 0.1768, 0.1430]
    train_acc = [0.1578, 0.5373, 0.6310, 0.6938, 0.7503]
    val_f1 = [0.7572, 0.8251, 0.8287, 0.8399, 0.8428]
    val_acc = [0.4720, 0.5720, 0.5790, 0.6040, 0.6130]


def chart_4_4_all_baseline_metrics():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        (train_loss, 'Training Loss', '#e15759'),
        (train_acc, 'Training Accuracy', '#59a14f'),
        (val_f1, 'Validation Macro F1', '#4e79a7'),
        (val_acc, 'Validation Accuracy', '#f28e2b'),
    ]
    for ax, (data, title, color) in zip(axes.flat, metrics):
        ax.plot(epochs, data, 'o-', color=color, linewidth=2, markersize=7)
        ax.annotate(f'{data[0]:.4f}', (epochs[0], data[0]),
                    textcoords="offset points", xytext=(-5, 10), fontsize=8, color=color)
        ax.annotate(f'{data[-1]:.4f}', (epochs[-1], data[-1]),
                    textcoords="offset points", xytext=(-5, 10), fontsize=8, color=color)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_xticks(epochs)

    fig.suptitle('Baseline Model Training Metrics (DistilBERT)', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/chart_4_4_all_baseline_metrics.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] CHART-4.4 saved")


def chart_4_5_dual_axis():
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color='#e15759')
    ax1.plot(epochs, train_loss, 'o-', color='#e15759', linewidth=2, markersize=7, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor='#e15759')
    ax1.set_xticks(epochs)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Macro F1', color='#4e79a7')
    ax2.plot(epochs, val_f1, 's-', color='#4e79a7', linewidth=2, markersize=7, label='Val Macro F1')
    ax2.tick_params(axis='y', labelcolor='#4e79a7')
    ax2.spines['top'].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc='center right')

    ax1.set_title('Training Loss vs Validation F1 (Baseline)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/chart_4_5_dual_axis.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] CHART-4.5 saved")


def chart_4_6_model_comparison():
    metrics = ['Macro F1', 'Micro F1', 'Accuracy', 'Hamming\nLoss']
    baseline_vals = [0.8428, 0.87, 0.613, 0.065]
    improved_vals = [0.91, 0.93, 0.78, 0.038]

    x = np.arange(len(metrics))
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, baseline_vals, width, label='DistilBERT (Kaggle)', color='#b0b0b0')
    ax.bar(x + width/2, improved_vals, width, label='RoBERTa (Custom)', color='#4e79a7')

    for i in range(len(metrics)):
        ax.text(i - width/2, baseline_vals[i] + 0.015, f'{baseline_vals[i]:.3f}', ha='center', fontsize=9)
        ax.text(i + width/2, improved_vals[i] + 0.015, f'{improved_vals[i]:.3f}', ha='center', fontsize=9)

    ax.set_ylabel('Score')
    ax.set_title('Model Performance: Baseline vs Improved', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/chart_4_6_model_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] CHART-4.6 saved")


def table_4_3_baseline_metrics():
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis('off')
    headers = ['Epoch', 'Train Loss', 'Train Accuracy', 'Val Macro F1', 'Val Accuracy']
    data = [[str(i+1), f'{train_loss[i]:.4f}', f'{train_acc[i]:.4f}',
             f'{val_f1[i]:.4f}', f'{val_acc[i]:.4f}'] for i in range(5)]

    table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center',
                     colWidths=[0.1, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('white')
        table[(0, j)].set_text_props(color='#333333', fontweight='bold')
    for i in range(1, 6):
        for j in range(len(headers)):
            table[(i, j)].set_facecolor('white')
            table[(i, j)].set_edgecolor('black')
    ax.set_title('TABLE 4.3: Baseline Model (DistilBERT) Training Metrics', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/table_4_3_baseline_metrics.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] TABLE-4.3 saved")


def table_4_4_architecture_comparison():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis('off')
    headers = ['Aspect', 'DistilBERT (Baseline)', 'RoBERTa-base (Improved)']
    data = [
        ['Parameters', '66M', '125M'],
        ['Hidden Size', '768', '768'],
        ['Layers', '6', '12'],
        ['Max Sequence Length', '256', '512'],
        ['Training Dataset', 'Kaggle (imbalanced)', 'Custom Scraped (balanced)'],
        ['Categories', '8 (original)', '8 (consolidated)'],
        ['Threshold', 'Fixed 0.5', 'Optuna per-category'],
        ['Best Macro F1', '0.8428', '~0.91'],
    ]
    table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center',
                     colWidths=[0.28, 0.34, 0.34])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('white')
        table[(0, j)].set_text_props(color='#333333', fontweight='bold')
    for i in range(1, len(data) + 1):
        for j in range(len(headers)):
            table[(i, j)].set_facecolor('white')
            table[(i, j)].set_edgecolor('black')
    ax.set_title('TABLE 4.4: Model Architecture Comparison', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/table_4_4_architecture_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] TABLE-4.4 saved")


if __name__ == '__main__':
    print("\n=== Training Metrics Charts ===\n")
    chart_4_4_all_baseline_metrics()
    chart_4_5_dual_axis()
    chart_4_6_model_comparison()
    table_4_3_baseline_metrics()
    table_4_4_architecture_comparison()
    print(f"\nAll saved to {OUT_DIR}/")
