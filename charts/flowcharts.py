"""
Generate additional flowchart / DFD / block diagrams for Chapter 6.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'text.color': '#333333',
    'font.family': 'sans-serif',
    'font.size': 10,
})

OUT_DIR = 'charts/output'
os.makedirs(OUT_DIR, exist_ok=True)


def fig_6_2_classification_flowchart():
    """Data Flow Diagram for Classification Pipeline."""
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')

    steps = [
        (6, 13.2, 'User Input\n(Text / URL)', '#4e79a7', 'rounded'),
        (6, 11.8, 'Text Cleaning\n& Preprocessing', '#6892b8', 'box'),
        (6, 10.4, 'RoBERTa Tokenizer\n(max 512 tokens)', '#76b7b2', 'box'),
        (6, 9.0,  'RoBERTa-base\nModel Inference', '#59a14f', 'box'),
        (6, 7.6,  'Sigmoid Activation\n(8 probabilities)', '#edc948', 'box'),
        (6, 6.2,  'Per-Category\nThreshold Check', '#f28e2b', 'diamond'),
        (3, 4.8,  'Label\nAssigned', '#59a14f', 'box'),
        (9, 4.8,  'Label\nSkipped', '#e15759', 'box'),
        (6, 3.4,  'Credibility\nFeature Extraction', '#b07aa1', 'box'),
        (6, 2.0,  'Weighted Score\nComputation', '#76b7b2', 'box'),
        (6, 0.6,  'Final Output\n(Labels + Score + XAI)', '#4e79a7', 'rounded'),
    ]

    for x, y, text, color, shape in steps:
        if shape == 'diamond':
            diamond = plt.Polygon([(x, y+0.55), (x+1.2, y), (x, y-0.55), (x-1.2, y)],
                                   facecolor=color, edgecolor='white', alpha=0.85)
            ax.add_patch(diamond)
            ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        elif shape == 'rounded':
            rect = mpatches.FancyBboxPatch((x-1.5, y-0.45), 3.0, 0.9,
                                            boxstyle="round,pad=0.15",
                                            facecolor=color, edgecolor='white', linewidth=2, alpha=0.85)
            ax.add_patch(rect)
            ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        else:
            rect = mpatches.FancyBboxPatch((x-1.5, y-0.45), 3.0, 0.9,
                                            boxstyle="round,pad=0.08",
                                            facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.85)
            ax.add_patch(rect)
            ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    # Arrows between main flow
    arrow_pairs = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(5,7),(8,9),(9,10)]
    positions = [(s[0], s[1]) for s in steps]
    for fr, to in arrow_pairs:
        x1, y1 = positions[fr]
        x2, y2 = positions[to]
        ax.annotate('', xy=(x2, y2+0.5), xytext=(x1, y1-0.5),
                    arrowprops=dict(arrowstyle='->', color='#888', lw=1.5))

    # Yes/No labels for diamond
    ax.text(4.2, 5.4, 'P >= T', fontsize=8, color='#59a14f', fontweight='bold')
    ax.text(7.8, 5.4, 'P < T', fontsize=8, color='#e15759', fontweight='bold')

    # Connect both branches back to credibility
    ax.annotate('', xy=(5, 3.85), xytext=(3, 4.35),
                arrowprops=dict(arrowstyle='->', color='#888', lw=1))
    ax.annotate('', xy=(7, 3.85), xytext=(9, 4.35),
                arrowprops=dict(arrowstyle='->', color='#888', lw=1))

    ax.set_title('Classification Pipeline Flowchart', fontsize=13, fontweight='bold', pad=10)
    plt.savefig(f'{OUT_DIR}/fig_6_2_classification_flowchart.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] FIG-6.2 saved")


def fig_6_3_credibility_workflow():
    """Block diagram for credibility scoring workflow."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Input
    rect = mpatches.FancyBboxPatch((5.5, 6.8), 3, 0.8, boxstyle="round,pad=0.12",
                                    facecolor='#4e79a7', edgecolor='white', linewidth=2, alpha=0.85)
    ax.add_patch(rect)
    ax.text(7, 7.2, 'Article Text + Title', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Feature boxes
    features = [
        (1, 5, 'Subjectivity\n(TextBlob)'),
        (3, 5, 'ALL-CAPS\nRatio'),
        (5, 5, 'Exclamation\nDensity'),
        (7, 5, 'Clickbait\nSignals (11)'),
        (9, 5, 'Sensational\nVocab (18)'),
        (11, 5, 'Article\nLength'),
        (13, 5, 'Headline\nQuestions'),
    ]
    for x, y, text in features:
        rect = mpatches.FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8, boxstyle="round,pad=0.06",
                                        facecolor='#76b7b2', edgecolor='white', alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=7, fontweight='bold', color='white')
        ax.annotate('', xy=(x, y+0.4), xytext=(7, 6.8),
                    arrowprops=dict(arrowstyle='->', color='#ccc', lw=0.8))

    # Source credibility (separate)
    rect = mpatches.FancyBboxPatch((6.2, 3.5), 1.6, 0.8, boxstyle="round,pad=0.06",
                                    facecolor='#f28e2b', edgecolor='white', alpha=0.8)
    ax.add_patch(rect)
    ax.text(7, 3.9, 'Source Domain\nLookup', ha='center', va='center', fontsize=7, fontweight='bold', color='white')

    # Aggregation
    rect = mpatches.FancyBboxPatch((5, 2.2), 4, 0.8, boxstyle="round,pad=0.12",
                                    facecolor='#59a14f', edgecolor='white', linewidth=2, alpha=0.85)
    ax.add_patch(rect)
    ax.text(7, 2.6, 'Weighted Aggregation\n& Normalization', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Arrows to aggregation
    for x, _, _ in features:
        ax.annotate('', xy=(7, 3.0), xytext=(x, 4.6),
                    arrowprops=dict(arrowstyle='->', color='#ccc', lw=0.6))
    ax.annotate('', xy=(7, 3.0), xytext=(7, 3.5),
                arrowprops=dict(arrowstyle='->', color='#f28e2b', lw=1.2))

    # Output
    rect = mpatches.FancyBboxPatch((5, 0.8), 4, 0.8, boxstyle="round,pad=0.12",
                                    facecolor='#4e79a7', edgecolor='white', linewidth=2, alpha=0.85)
    ax.add_patch(rect)
    ax.text(7, 1.2, 'Credibility Score (0-100)', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax.annotate('', xy=(7, 1.6), xytext=(7, 2.2),
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.5))

    # Tier labels
    ax.text(3, 1.2, 'Low (0-50)', fontsize=9, color='#e15759', fontweight='bold')
    ax.text(10, 1.2, 'High (75-100)', fontsize=9, color='#59a14f', fontweight='bold')

    ax.set_title('Credibility Scoring Workflow', fontsize=13, fontweight='bold', pad=10)
    plt.savefig(f'{OUT_DIR}/fig_6_3_credibility_workflow.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] FIG-6.3 saved")


def fig_6_4_deployment_dfd():
    """DFD for the deployment architecture."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # User
    circle = plt.Circle((2, 5.5), 0.7, facecolor='#4e79a7', edgecolor='white', linewidth=2, alpha=0.85)
    ax.add_patch(circle)
    ax.text(2, 5.5, 'User', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Streamlit
    rect = mpatches.FancyBboxPatch((4.5, 4.8), 2.5, 1.4, boxstyle="round,pad=0.12",
                                    facecolor='#e15759', edgecolor='white', linewidth=2, alpha=0.85)
    ax.add_patch(rect)
    ax.text(5.75, 5.5, 'Streamlit\nFrontend', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # FastAPI
    rect = mpatches.FancyBboxPatch((8.5, 4.8), 2.5, 1.4, boxstyle="round,pad=0.12",
                                    facecolor='#f28e2b', edgecolor='white', linewidth=2, alpha=0.85)
    ax.add_patch(rect)
    ax.text(9.75, 5.5, 'FastAPI\nBackend', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Model components
    components = [
        (8, 2.5, 'RoBERTa\nClassifier', '#59a14f'),
        (10, 2.5, 'Credibility\nScorer', '#76b7b2'),
        (12, 2.5, 'XAI\nModule', '#edc948'),
        (12, 4.5, 'Feedback\nDB', '#b07aa1'),
    ]
    for x, y, text, color in components:
        rect = mpatches.FancyBboxPatch((x-0.9, y-0.5), 1.8, 1.0, boxstyle="round,pad=0.08",
                                        facecolor=color, edgecolor='white', alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=7, fontweight='bold', color='white')

    # Arrows
    ax.annotate('', xy=(4.5, 5.5), xytext=(2.7, 5.5),
                arrowprops=dict(arrowstyle='->', color='#888', lw=2))
    ax.text(3.5, 5.8, 'Text / URL', fontsize=7, color='#888')

    ax.annotate('', xy=(8.5, 5.5), xytext=(7.0, 5.5),
                arrowprops=dict(arrowstyle='->', color='#888', lw=2))
    ax.text(7.4, 5.8, 'API Call', fontsize=7, color='#888')

    ax.annotate('', xy=(7.0, 5.2), xytext=(8.5, 5.2),
                arrowprops=dict(arrowstyle='->', color='#59a14f', lw=1.5))
    ax.text(7.3, 4.85, 'JSON Response', fontsize=7, color='#59a14f')

    for x, y, _, _ in components[:3]:
        ax.annotate('', xy=(x, y+0.5), xytext=(9.75, 4.8),
                    arrowprops=dict(arrowstyle='->', color='#ccc', lw=1))

    ax.annotate('', xy=(12, 3.5), xytext=(12, 4.0),
                arrowprops=dict(arrowstyle='<->', color='#b07aa1', lw=1))

    ax.set_title('Deployment Architecture - Data Flow Diagram', fontsize=13, fontweight='bold', pad=10)
    plt.savefig(f'{OUT_DIR}/fig_6_4_deployment_dfd.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] FIG-6.4 saved")


if __name__ == '__main__':
    fig_6_2_classification_flowchart()
    fig_6_3_credibility_workflow()
    fig_6_4_deployment_dfd()
    print("All flowcharts saved.")
