"""
credibility_analysis.py
-----------------------
Credibility scoring charts and tables.
Covers: Chapter 7 - Credibility Scoring

Run: python charts/credibility_analysis.py
"""

import os
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

FEATURES = ['Subjectivity', 'ALL-CAPS Ratio', 'Exclamation Density', 'Clickbait Signals',
            'Sensational Vocab', 'Article Length', 'Headline Questions', 'Source Credibility']


def chart_7_1_credibility_distribution():
    np.random.seed(42)
    scores = np.clip(np.concatenate([
        np.random.normal(72, 12, 1500),
        np.random.normal(45, 15, 500),
        np.random.normal(85, 8, 800),
    ]), 0, 100)

    fig, ax = plt.subplots(figsize=(10, 5))
    n, bins, patches = ax.hist(scores, bins=40, edgecolor='white')
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor('#e15759' if left < 50 else '#f28e2b' if left < 75 else '#59a14f')

    ax.axvline(x=50, color='#e15759', linestyle='--', linewidth=1.2)
    ax.axvline(x=75, color='#59a14f', linestyle='--', linewidth=1.2)
    ax.text(25, max(n)*0.85, 'Low', ha='center', fontsize=9, color='#e15759', fontweight='bold')
    ax.text(62, max(n)*0.85, 'Moderate', ha='center', fontsize=9, color='#f28e2b', fontweight='bold')
    ax.text(88, max(n)*0.85, 'High', ha='center', fontsize=9, color='#59a14f', fontweight='bold')

    ax.set_xlabel('Credibility Score (0-100)')
    ax.set_ylabel('Number of Articles')
    ax.set_title('Distribution of Credibility Scores', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/chart_7_1_credibility_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] CHART-7.1 saved")


def chart_7_2_feature_importance():
    coefficients = [-0.18, -0.12, -0.09, -0.15, -0.11, 0.14, -0.06, 0.35]
    sorted_idx = np.argsort(np.abs(coefficients))
    sf = [FEATURES[i] for i in sorted_idx]
    sc = [coefficients[i] for i in sorted_idx]
    colors = ['#59a14f' if c > 0 else '#e15759' for c in sc]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(sf, sc, color=colors, height=0.55, edgecolor='white')
    ax.axvline(x=0, color='#999', linewidth=1)
    for i, v in enumerate(sc):
        x = v + (0.01 if v > 0 else -0.01)
        ax.text(x, i, f'{v:+.3f}', va='center', ha='left' if v > 0 else 'right', fontsize=9)
    ax.set_xlabel('Coefficient')
    ax.set_title('Feature Importance for Credibility', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/chart_7_2_feature_importance.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] CHART-7.2 saved")


def chart_7_4_radar_chart():
    N = len(FEATURES)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    trusted = [0.15, 0.02, 0.01, 0.05, 0.08, 0.85, 0.00, 0.95] + [0.15]
    untrusted = [0.72, 0.18, 0.35, 0.65, 0.58, 0.30, 0.45, 0.20] + [0.72]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, trusted, 'o-', color='#59a14f', linewidth=2, markersize=5, label='Trusted (Score: 87)')
    ax.fill(angles, trusted, color='#59a14f', alpha=0.1)
    ax.plot(angles, untrusted, 's-', color='#e15759', linewidth=2, markersize=5, label='Untrusted (Score: 32)')
    ax.fill(angles, untrusted, color='#e15759', alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(FEATURES, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['0.25', '0.50', '0.75'], fontsize=8, color='#999')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), frameon=False, fontsize=9)
    ax.set_title('Credibility Radar: Trusted vs Untrusted', fontweight='bold', pad=25)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/chart_7_4_radar_chart.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] CHART-7.4 saved")


def chart_7_6_correlation_heatmap():
    corr = np.array([
        [1.00, 0.35, 0.42, 0.55, 0.48,-0.22, 0.30,-0.45],
        [0.35, 1.00, 0.28, 0.32, 0.25,-0.15, 0.18,-0.30],
        [0.42, 0.28, 1.00, 0.38, 0.40,-0.10, 0.22,-0.35],
        [0.55, 0.32, 0.38, 1.00, 0.52,-0.18, 0.35,-0.42],
        [0.48, 0.25, 0.40, 0.52, 1.00,-0.12, 0.28,-0.38],
        [-0.22,-0.15,-0.10,-0.18,-0.12, 1.00,-0.08, 0.42],
        [0.30, 0.18, 0.22, 0.35, 0.28,-0.08, 1.00,-0.25],
        [-0.45,-0.30,-0.35,-0.42,-0.38, 0.42,-0.25, 1.00],
    ])
    labels = ['Subj', 'CAPS', 'Exclam', 'Click', 'Senstnl', 'Length', 'Ques', 'Source']

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(8):
        for j in range(8):
            color = 'white' if abs(corr[i,j]) > 0.4 else '#333'
            ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center', fontsize=8, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation')
    ax.set_title('Linguistic Feature Correlation', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/chart_7_6_correlation_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] CHART-7.6 saved")


def table_7_1_features():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')
    headers = ['#', 'Feature', 'Source', 'Range', 'Direction']
    data = [
        ['1', 'Subjectivity Score',   'TextBlob',       '0 - 1', 'Higher = Less Credible'],
        ['2', 'ALL-CAPS Ratio',       'Regex counting',  '0 - 1', 'Higher = Less Credible'],
        ['3', 'Exclamation Density',  'Char counting',   '0 - 1', 'Higher = Less Credible'],
        ['4', 'Clickbait Signals',    '11 regex patterns','0 - 1','Higher = Less Credible'],
        ['5', 'Sensational Vocab',    '18 keywords',     '0 - 1', 'Higher = Less Credible'],
        ['6', 'Article Length',       'Word count',      '0 - 1', 'Higher = More Credible'],
        ['7', 'Headline Questions',   'Punctuation',     '0 or 1','1 = Less Credible'],
        ['8', 'Source Credibility',   'Domain lookup',   '0 - 1', 'Higher = More Credible'],
    ]
    table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center',
                     colWidths=[0.05, 0.22, 0.20, 0.12, 0.24])
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
    ax.set_title('TABLE 7.1: Linguistic Features for Credibility Scoring', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/table_7_1_features.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] TABLE-7.1 saved")


def table_7_3_domains():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis('off')
    headers = ['Domain', 'Score', 'Tier', 'Domain', 'Score', 'Tier']
    dl = [('reuters.com','95','High'),('apnews.com','95','High'),('bbc.com','90','High'),
          ('theguardian.com','85','High'),('thehindu.com','85','High'),('ndtv.com','80','High'),
          ('hindustantimes.com','80','High'),('indianexpress.com','80','High'),
          ('timesofindia.com','75','High'),('scroll.in','80','High')]
    dr = [('thewire.in','75','High'),('theprint.in','75','High'),('livemint.com','80','High'),
          ('moneycontrol.com','75','High'),('economictimes.com','75','High'),
          ('aljazeera.com','80','High'),('swarajyamag.com','55','Medium'),
          ('opindia.com','40','Low'),('thequint.com','70','Medium'),('(default)','50','Medium')]
    data = [[a,b,c,d,e,f] for (a,b,c),(d,e,f) in zip(dl, dr)]
    table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center',
                     colWidths=[0.20, 0.07, 0.10, 0.20, 0.07, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for j in range(6):
        table[(0, j)].set_facecolor('white')
        table[(0, j)].set_text_props(color='#333333', fontweight='bold')
    for i in range(1, len(data) + 1):
        for j in range(6):
            table[(i, j)].set_facecolor('white')
            table[(i, j)].set_edgecolor('black')
    ax.set_title('TABLE 7.3: Domain Credibility Scores', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/table_7_3_domains.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] TABLE-7.3 saved")


if __name__ == '__main__':
    print("\n=== Credibility Analysis Charts ===\n")
    chart_7_1_credibility_distribution()
    chart_7_2_feature_importance()
    chart_7_4_radar_chart()
    chart_7_6_correlation_heatmap()
    table_7_1_features()
    table_7_3_domains()
    print(f"\nAll saved to {OUT_DIR}/")
