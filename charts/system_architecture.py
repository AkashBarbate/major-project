"""
system_architecture.py
----------------------
Architecture diagrams, labeling charts, and system design tables.
Covers: Chapter 1, 3, 9

Run: python charts/system_architecture.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

CATEGORIES = ['Politics', 'Sci & Tech', 'Business\n& Economy', "Int'l\nAffairs",
              'Education', 'Health', 'Sports', 'Entertainment']
COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7']


def fig_1_1_architecture():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16); ax.set_ylim(0, 7)
    ax.axis('off')

    boxes = [
        (0.3, 4.8, 'Data\nCollection', '#4e79a7', 'RSS + CC-News\nAG News + MIND'),
        (3.0, 4.8, 'Pre-\nprocessing', '#6892b8', 'Clean + Dedup\n+ Capping'),
        (5.7, 4.8, 'LLM\nLabeling', '#76b7b2', 'Gemini API\n+ DeBERTa NLI'),
        (8.4, 4.8, 'Model\nTraining', '#59a14f', 'RoBERTa + Optuna\n+ Calibration'),
        (11.1, 4.8, 'FastAPI\nBackend', '#f28e2b', 'Classify + Cred\n+ XAI + Feedback'),
        (13.8, 4.8, 'Streamlit\nFrontend', '#e15759', 'URL/Text input\nVisualization'),
    ]
    for x, y, title, color, detail in boxes:
        rect = mpatches.FancyBboxPatch((x, y-1.2), 2.2, 2.4, boxstyle="round,pad=0.12",
                                        facecolor=color, edgecolor='white', linewidth=2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x+1.1, y+0.15, title, ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        ax.text(x+1.1, y-0.75, detail, ha='center', va='center', fontsize=7, color='white')

    for i in range(5):
        ax.annotate('', xy=(boxes[i+1][0], boxes[i][1]-0.0),
                    xytext=(boxes[i][0]+2.2, boxes[i][1]),
                    arrowprops=dict(arrowstyle='->', color='#888', lw=1.5))

    bottom = [
        (2.5, 1.2, 'Credibility\nModule', '#76b7b2', 'Ridge + 8 Features'),
        (6.0, 1.2, 'XAI Module', '#f28e2b', 'Integrated Gradients'),
        (9.5, 1.2, 'KeyBERT', '#edc948', 'Topic Keywords'),
        (13.0, 1.2, 'Feedback DB', '#b07aa1', 'SQLite + Active Learning'),
    ]
    for x, y, title, color, detail in bottom:
        rect = mpatches.FancyBboxPatch((x, y-0.7), 2.2, 1.4, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='white', linewidth=1, alpha=0.6)
        ax.add_patch(rect)
        ax.text(x+1.1, y+0.1, title, ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        ax.text(x+1.1, y-0.35, detail, ha='center', va='center', fontsize=7, color='white')

    for bx, _, _, _, _ in bottom:
        ax.annotate('', xy=(12.2, 3.6), xytext=(bx+1.1, 1.9),
                    arrowprops=dict(arrowstyle='->', color='#ccc', lw=0.8, linestyle='--'))

    ax.set_title('System Architecture - News Classification Pipeline', fontsize=13, fontweight='bold', pad=10)
    plt.savefig(f'{OUT_DIR}/fig_1_1_architecture.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] FIG-1.1 saved")


def fig_1_3_category_consolidation():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 11)
    ax.axis('off')

    old = ['Politics', 'Technology', 'Business', 'Economy', 'International_Affairs',
           'Education', 'Health', 'Science/Other', 'Sports', 'Entertainment']
    new = ['Politics', 'Science and Technology', 'Business and Economy',
           'International Affairs', 'Education', 'Health', 'Sports', 'Entertainment']

    ax.text(2.2, 10.5, 'Original (10)', ha='center', fontsize=11, fontweight='bold', color='#888')
    for i, cat in enumerate(old):
        y = 9.8 - i * 0.95
        rect = mpatches.FancyBboxPatch((0.4, y-0.25), 3.6, 0.5, boxstyle="round,pad=0.08",
                                        facecolor='#f0f0f0', edgecolor='#ccc')
        ax.add_patch(rect)
        ax.text(2.2, y, cat, ha='center', va='center', fontsize=8)

    ax.text(9.8, 10.5, 'Final (8)', ha='center', fontsize=11, fontweight='bold', color='#4e79a7')
    for i, (cat, color) in enumerate(zip(new, COLORS)):
        y = 9.8 - i * 1.15
        rect = mpatches.FancyBboxPatch((8.0, y-0.3), 3.6, 0.6, boxstyle="round,pad=0.08",
                                        facecolor=color, edgecolor='white', alpha=0.8)
        ax.add_patch(rect)
        ax.text(9.8, y, cat, ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    mappings = [(0,0),(1,1),(2,2),(3,2),(4,3),(5,4),(6,5),(7,1),(8,6),(9,7)]
    for oi, ni in mappings:
        ys = 9.8 - oi * 0.95
        ye = 9.8 - ni * 1.15
        lw = 2 if (oi,ni) in [(3,2),(7,1)] else 0.8
        color = '#e15759' if (oi,ni) in [(3,2),(7,1)] else '#ccc'
        ax.annotate('', xy=(8.0, ye), xytext=(4.0, ys),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw, connectionstyle='arc3,rad=0.08'))

    ax.text(6, 4.5, 'MERGED', ha='center', fontsize=9, color='#e15759', fontweight='bold', fontstyle='italic')
    ax.set_title('Category Consolidation (10 -> 8)', fontsize=13, fontweight='bold', pad=10)
    plt.savefig(f'{OUT_DIR}/fig_1_3_category_consolidation.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] FIG-1.3 saved")


def chart_3_1_label_distribution():
    cats = ['Sports', 'Health', 'Politics', 'Sci & Tech',
            'Business & Eco', "Int'l Affairs", 'Entertainment', 'Education']
    counts = [420, 395, 385, 370, 365, 340, 335, 310]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(cats, counts, color='#4e79a7', height=0.55, edgecolor='white')
    for bar, c in zip(bars, counts):
        pct = c / sum(counts) * 100
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                f'{c} ({pct:.1f}%)', va='center', fontsize=9, color='#555')
    ax.set_xlabel('Number of Articles')
    ax.set_title('Label Distribution After Gemini Labeling (8 Categories)', fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts)*1.18)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/chart_3_1_label_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] CHART-3.1 saved")


def chart_3_3_cooccurrence():
    labels = ['Pol', 'S&T', 'B&E', 'IntA', 'Edu', 'Health', 'Sports', 'Ent']
    cooccur = np.array([
        [385,25,35,85,10,5,3,2],[25,370,20,8,15,45,2,5],[35,20,365,40,3,5,2,8],
        [85,8,40,340,12,5,3,2],[10,15,3,12,310,18,2,10],[5,45,5,5,18,395,2,3],
        [3,2,2,3,2,2,420,8],[2,5,8,2,10,3,8,335],
    ])
    display = cooccur.astype(float).copy()
    np.fill_diagonal(display, np.nan)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(display, cmap='YlOrRd')
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(8):
        for j in range(8):
            if i != j:
                color = 'white' if cooccur[i,j] > 30 else '#333'
                ax.text(j, i, str(cooccur[i,j]), ha='center', va='center', fontsize=8, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Co-occurrence Count')
    ax.set_title('Multi-Label Co-occurrence Heatmap', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/chart_3_3_cooccurrence.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] CHART-3.3 saved")


def table_3_1_labeling_models():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    headers = ['Model', 'Type', 'RPD Limit', 'RPM', 'Schema', 'Priority']
    data = [
        ['gemini-3.1-flash-lite','Gemini','500','15','Pydantic','1 (Primary)'],
        ['gemma-4-31b-it','Gemma','1,500','15','JSON prompt','2'],
        ['gemma-3-27b-it','Gemma','14,400','30','JSON prompt','3'],
        ['gemini-2.5-flash','Gemini','20','5','Pydantic','4'],
        ['gemini-2.5-flash-lite','Gemini','20','10','Pydantic','5'],
    ]
    table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center',
                     colWidths=[0.24, 0.09, 0.11, 0.07, 0.16, 0.14])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('white')
        table[(0, j)].set_text_props(color='#333333', fontweight='bold')
    for i in range(1, len(data) + 1):
        for j in range(len(headers)):
            bg = 'white'
            table[(i, j)].set_facecolor(bg)
            table[(i, j)].set_edgecolor('black')
    ax.set_title('TABLE 3.1: Labeling Models (Gemini/Gemma)', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/table_3_1_labeling_models.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] TABLE-3.1 saved")


def table_9_1_api_endpoints():
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis('off')
    headers = ['Method', 'Endpoint', 'Description', 'Response']
    data = [
        ['POST','/predict','Classify text input','Labels + Credibility + XAI'],
        ['POST','/predict-url','Scrape URL and classify','Labels + Credibility + XAI'],
        ['GET','/health','API health check','Status + Models loaded'],
        ['GET','/categories','List all categories','List of 8 categories'],
        ['POST','/feedback','Submit correction','Feedback ID'],
        ['GET','/feedback/count','Count corrections','Total count'],
        ['GET','/feedback/export','Export corrections CSV','CSV download'],
    ]
    table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center',
                     colWidths=[0.08, 0.18, 0.30, 0.30])
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
    ax.set_title('TABLE 9.1: FastAPI REST Endpoints', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/table_9_1_api_endpoints.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] TABLE-9.1 saved")


if __name__ == '__main__':
    print("\n=== Architecture & System Charts ===\n")
    fig_1_1_architecture()
    fig_1_3_category_consolidation()
    chart_3_1_label_distribution()
    chart_3_3_cooccurrence()
    table_3_1_labeling_models()
    table_9_1_api_endpoints()
    print(f"\nAll saved to {OUT_DIR}/")
