"""
Generate the NewsLens PPT with all slides, content, and embedded charts.
Run: python ppt/generate_ppt.py
Output: ppt/NewsLens_Presentation.pptx
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

CHART_DIR = os.path.join(os.path.dirname(__file__), '..', 'charts', 'output')
OUT_DIR = os.path.dirname(__file__)

# Colors
NAVY = RGBColor(0x16, 0x21, 0x3e)
DARK_BG = RGBColor(0x1a, 0x1a, 0x2e)
ACCENT = RGBColor(0x4e, 0x79, 0xa7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xCC, 0xCC, 0xCC)
SOFT_BLUE = RGBColor(0x76, 0xb7, 0xb2)
GREEN = RGBColor(0x59, 0xa1, 0x4f)
ORANGE = RGBColor(0xf2, 0x8e, 0x2b)


def set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_title(slide, text, left=0.5, top=0.3, width=9, size=28, color=WHITE, bold=True):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(0.8))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = 'Calibri'
    return txBox


def add_text_box(slide, text, left, top, width, height, size=18, color=WHITE, bold=False, align=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, line in enumerate(text.split('\n')):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.font.bold = bold
        p.font.name = 'Calibri'
        p.alignment = align
        p.space_after = Pt(4)
    return txBox


def add_bullets(slide, items, left, top, width, height, size=17, color=WHITE):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.font.name = 'Calibri'
        p.space_before = Pt(6)
        p.space_after = Pt(4)
        p.level = 0
        # Add bullet character
        run = p.runs[0] if p.runs else p.add_run()
        run.text = "\u2022  " + item
        p.clear()
        r = p.add_run()
        r.text = "\u2022  " + item
        r.font.size = Pt(size)
        r.font.color.rgb = color
        r.font.name = 'Calibri'
    return txBox


def add_image_safe(slide, img_name, left, top, width, height=None):
    img_path = os.path.join(CHART_DIR, img_name)
    if os.path.exists(img_path):
        if height:
            slide.shapes.add_picture(img_path, Inches(left), Inches(top), Inches(width), Inches(height))
        else:
            slide.shapes.add_picture(img_path, Inches(left), Inches(top), Inches(width))
    else:
        add_text_box(slide, f"[IMAGE: {img_name}]", left, top, width, 0.5, size=12, color=LIGHT_GRAY)


def add_accent_line(slide, left, top, width):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Pt(3))
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT
    shape.line.fill.background()


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # ===== SLIDE 1: TITLE =====
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    set_slide_bg(slide, DARK_BG)
    add_accent_line(slide, 1, 1.2, 11.3)
    add_text_box(slide, "AUTOMATED NEWS CLASSIFICATION", 1, 1.5, 11.3, 1, size=36, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_text_box(slide, "Multi-Label Classification with Credibility Scoring & Explainable AI", 1, 2.3, 11.3, 0.6, size=20, color=SOFT_BLUE, align=PP_ALIGN.CENTER)
    add_accent_line(slide, 4, 3.2, 5.3)
    add_text_box(slide, "Akash Barbate (23U02030)\nMarrapu Sai Varun (23U02065)", 1, 3.6, 11.3, 0.9, size=18, color=WHITE, align=PP_ALIGN.CENTER)
    add_text_box(slide, "Under the supervision of\nDr. Gagan Vishwakarma, Assistant Professor, CSE", 1, 4.6, 11.3, 0.8, size=16, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)
    add_text_box(slide, "Indian Institute of Information Technology, Bhopal\nApril 2026", 1, 5.8, 11.3, 0.8, size=16, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)

    # ===== SLIDE 2: INTRODUCTION =====
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, NAVY)
    add_title(slide, "Introduction")
    add_accent_line(slide, 0.5, 1.0, 12.3)
    add_bullets(slide, [
        "News articles often span multiple topics \u2014 single-label systems miss relevant categories",
        "No existing platform provides credibility assessment alongside classification",
        "NewsLens: end-to-end pipeline for multi-label classification + credibility scoring + XAI",
        "Covers 8 categories: Politics, Sci & Tech, Business, Int'l Affairs, Education, Health, Sports, Entertainment",
        "Deployed as FastAPI backend + Streamlit frontend for real-time analysis",
    ], 0.5, 1.3, 5.8, 4.5, size=16)
    add_image_safe(slide, 'fig_1_1_architecture.png', 6.8, 1.5, 6, 4)

    # ===== SLIDE 3: LITERATURE REVIEW =====
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, NAVY)
    add_title(slide, "Background & Literature Review")
    add_accent_line(slide, 0.5, 1.0, 12.3)
    add_bullets(slide, [
        "Traditional ML (Naive Bayes, SVM) \u2014 relies on TF-IDF, ignores context",
        "Deep Learning (CNN, LSTM) \u2014 auto feature learning but sequential bottleneck",
        "Transformers (BERT, RoBERTa) \u2014 self-attention captures full context",
        "RoBERTa-base chosen: +8% macro F1 over DistilBERT on our dataset",
        "Research Gap: no system combines multi-label + credibility + XAI",
    ], 0.5, 1.3, 5.8, 4, size=16)
    add_image_safe(slide, 'table_4_4_architecture_comparison.png', 6.8, 1.5, 6, 3.5)

    # ===== SLIDE 4: PROBLEM STATEMENT =====
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, NAVY)
    add_title(slide, "Problem Statement & Objectives")
    add_accent_line(slide, 0.5, 1.0, 12.3)
    add_bullets(slide, [
        "Problem 1: Single-label classification hides articles from secondary categories",
        "Problem 2: No credibility signal \u2014 readers cannot judge source reliability",
        "Objective: Assign multiple labels AND a trustworthiness score (0\u2013100)",
        "Custom dataset from 460+ domains (not outdated Kaggle data)",
        "Per-category threshold optimization via Optuna (not uniform 0.5)",
        "Explainable AI for transparent, interpretable decisions",
    ], 0.5, 1.3, 6.2, 4.5, size=16)
    add_image_safe(slide, 'fig_1_3_category_consolidation.png', 7.2, 1.3, 5.5, 4.5)

    # ===== SLIDE 5: METHODOLOGY — DATA =====
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, NAVY)
    add_title(slide, "Methodology \u2014 Data Pipeline")
    add_accent_line(slide, 0.5, 1.0, 12.3)
    add_bullets(slide, [
        "Scraped 5,248 articles from RSS, CC-News, web scraping",
        "5-stage cleaning: domain capping \u2192 HTML removal \u2192 short article removal \u2192 stub filtering \u2192 deduplication",
        "Final dataset: 2,947 balanced, high-quality articles",
        "LLM annotation using Gemini API (5 model rotation for rate limits)",
        "Fallback: DeBERTa zero-shot for ambiguous articles",
    ], 0.5, 1.3, 5.5, 4, size=16)
    add_image_safe(slide, 'chart_2_7_data_funnel.png', 6.5, 1.2, 3.2, 2.5)
    add_image_safe(slide, 'chart_3_1_label_distribution.png', 9.8, 1.2, 3.2, 2.5)
    add_image_safe(slide, 'chart_3_3_cooccurrence.png', 7.5, 4.0, 4.5, 3.2)

    # ===== SLIDE 6: METHODOLOGY — TRAINING =====
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, NAVY)
    add_title(slide, "Methodology \u2014 Model Training")
    add_accent_line(slide, 0.5, 1.0, 12.3)
    add_bullets(slide, [
        "RoBERTa-base (125M params) fine-tuned for multi-label output",
        "Sigmoid activation + Binary Cross-Entropy loss (independent per label)",
        "Training: 5 epochs, lr = 2e-5, batch = 16, AdamW optimizer",
        "Baseline (DistilBERT): macro F1 = 0.8428",
        "Improved (RoBERTa + Optuna thresholds): macro F1 = 0.91",
        "Temperature calibration (T = 1.42): ECE reduced by 62%",
    ], 0.5, 1.3, 5.5, 4.5, size=16)
    add_image_safe(slide, 'chart_4_4_all_baseline_metrics.png', 6.5, 1.3, 6.3, 4.5)

    # ===== SLIDE 7: CREDIBILITY & XAI =====
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, NAVY)
    add_title(slide, "Credibility Scoring & Explainable AI")
    add_accent_line(slide, 0.5, 1.0, 12.3)
    add_bullets(slide, [
        "8 linguistic features: subjectivity, CAPS ratio, exclamation density, clickbait, sensational vocab, length, headline questions, source reputation",
        "Weighted formula \u2192 credibility score 0\u2013100",
        "XAI via Integrated Gradients (transformers-interpret)",
        "Per-word attribution for each predicted label",
        "Fully transparent \u2014 user sees WHY each label was assigned",
    ], 0.5, 1.3, 5.8, 4, size=16)
    add_image_safe(slide, 'chart_7_4_radar_chart.png', 7, 1.3, 5.8, 5)

    # ===== SLIDE 8: SYSTEM DIAGRAM =====
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, NAVY)
    add_title(slide, "System Architecture & Algorithm Flow")
    add_accent_line(slide, 0.5, 1.0, 12.3)
    add_image_safe(slide, 'fig_6_2_classification_flowchart.png', 1, 1.3, 5.5, 5.8)
    add_image_safe(slide, 'fig_6_3_credibility_workflow.png', 6.8, 1.3, 6, 4)
    add_text_box(slide, "Input \u2192 Clean \u2192 Tokenize \u2192 RoBERTa \u2192 Sigmoid \u2192 Threshold Check \u2192 Labels + Credibility + XAI",
                 1, 6.8, 11.5, 0.5, size=14, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)

    # ===== SLIDE 9: IMPLEMENTATION =====
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, NAVY)
    add_title(slide, "Implementation & Tools")
    add_accent_line(slide, 0.5, 1.0, 12.3)
    add_bullets(slide, [
        "Python 3.12 + PyTorch + Hugging Face Transformers",
        "Optuna: Bayesian threshold optimization (200 trials, TPE sampler)",
        "TextBlob for subjectivity, KeyBERT for keyword extraction",
        "FastAPI backend: 7 REST endpoints for classification, credibility, feedback",
        "Streamlit frontend: URL/text input, category cards, credibility gauge, radar chart",
        "SQLite feedback database for active learning loop",
    ], 0.5, 1.3, 5.5, 4.5, size=16)
    add_image_safe(slide, 'table_9_1_api_endpoints.png', 6.5, 1.3, 6.3, 3.5)
    add_image_safe(slide, 'table_8_1_xai_methods.png', 6.5, 4.9, 6.3, 2.3)

    # ===== SLIDE 10: RESULTS — CLASSIFICATION =====
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, NAVY)
    add_title(slide, "Results \u2014 Classification Performance")
    add_accent_line(slide, 0.5, 1.0, 12.3)
    add_image_safe(slide, 'table_10_2_per_category.png', 0.3, 1.3, 6.5, 3)
    add_image_safe(slide, 'chart_10_1_roc_curves.png', 7, 1.3, 5.8, 4.2)
    add_text_box(slide, "Macro F1: 0.91  |  Micro F1: 0.93  |  Subset Accuracy: 0.78  |  Hamming Loss: 0.038",
                 0.5, 6.5, 12, 0.5, size=18, color=SOFT_BLUE, bold=True, align=PP_ALIGN.CENTER)
    add_image_safe(slide, 'chart_10_2_pr_curves.png', 0.3, 4.5, 6.5, 2)

    # ===== SLIDE 11: RESULTS — THRESHOLDS =====
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, NAVY)
    add_title(slide, "Results \u2014 Threshold Optimization & Calibration")
    add_accent_line(slide, 0.5, 1.0, 12.3)
    add_image_safe(slide, 'chart_5_1_thresholds.png', 0.3, 1.3, 6.3, 3)
    add_image_safe(slide, 'chart_6_2_ece_comparison.png', 7, 1.3, 5.8, 3)
    add_bullets(slide, [
        "Thresholds range: 0.4783 (Business) to 0.7679 (Int'l Affairs)",
        "Per-category tuning improved F1 by ~5% over uniform 0.5",
        "Temperature scaling (T = 1.42): ECE dropped 0.082 \u2192 0.031 (62% reduction)",
    ], 0.5, 4.8, 12, 2.5, size=16)

    # ===== SLIDE 12: RESULTS — CREDIBILITY & XAI =====
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, NAVY)
    add_title(slide, "Results \u2014 Credibility & Explainability")
    add_accent_line(slide, 0.5, 1.0, 12.3)
    add_image_safe(slide, 'chart_7_2_feature_importance.png', 0.3, 1.3, 6, 3)
    add_image_safe(slide, 'fig_8_1_xai_keyword_impact.png', 6.8, 1.3, 6, 3.5)
    add_bullets(slide, [
        "Source credibility = strongest positive signal (coefficient 0.35)",
        "Subjectivity = strongest negative signal (coefficient -0.18)",
        "XAI confirms model uses semantically meaningful words per category",
        "\"parliament\" \u2192 Politics; \"cricket\" \u2192 Sports; \"vaccine\" \u2192 Health",
    ], 0.5, 5, 12, 2, size=16)

    # ===== SLIDE 13: CONCLUSION =====
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, NAVY)
    add_title(slide, "Conclusion & Future Scope")
    add_accent_line(slide, 0.5, 1.0, 12.3)

    add_text_box(slide, "Conclusion", 0.5, 1.3, 5.5, 0.5, size=22, color=SOFT_BLUE, bold=True)
    add_bullets(slide, [
        "Built end-to-end multi-label news classification with credibility + XAI",
        "Custom dataset: 2,947 articles from 460+ domains",
        "RoBERTa + Optuna achieves 0.91 macro F1",
        "Credibility module: transparent 0\u2013100 scoring using 8 features",
        "Deployed as real-time web application (FastAPI + Streamlit)",
    ], 0.5, 1.9, 5.5, 3.5, size=15)

    add_text_box(slide, "Future Scope", 7, 1.3, 5.5, 0.5, size=22, color=ORANGE, bold=True)
    add_bullets(slide, [
        "Fact-checking API integration for claim verification",
        "Multilingual support (Hindi, Marathi, regional languages)",
        "Hierarchical classification (Sports \u2192 Cricket \u2192 IPL)",
        "Active learning from user feedback (feedback DB already built)",
        "Mobile app deployment using quantized model",
    ], 7, 1.9, 5.5, 3.5, size=15)

    # ===== SLIDE 14: REFERENCES =====
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, NAVY)
    add_title(slide, "References")
    add_accent_line(slide, 0.5, 1.0, 12.3)

    refs_left = (
        "[1] Tsoumakas & Katakis \u2014 Multi-label Classification Overview (2007)\n"
        "[2] Zhang & Zhou \u2014 Multi-label Learning, IEEE TKDE (2014)\n"
        "[3] Liu et al. \u2014 RoBERTa: Robust BERT Pretraining (2019)\n"
        "[4] Devlin et al. \u2014 BERT: Pre-training Transformers (2019)\n"
        "[5] Vaswani et al. \u2014 Attention Is All You Need (2017)\n"
        "[6] Akiba et al. \u2014 Optuna Framework (2019)\n"
        "[7] Shu et al. \u2014 Fake News Detection Survey (2017)"
    )
    refs_right = (
        "[8] Vosoughi et al. \u2014 Spread of True & False News (2018)\n"
        "[9] Sundararajan et al. \u2014 Integrated Gradients (2017)\n"
        "[10] Guo et al. \u2014 Calibration of Neural Nets (2017)\n"
        "[11] Kim \u2014 CNN for Sentence Classification (2014)\n"
        "[12] Hochreiter & Schmidhuber \u2014 LSTM (1997)\n"
        "[13] Hugging Face Transformers Library\n"
        "[14] FastAPI & Streamlit Documentation"
    )
    add_text_box(slide, refs_left, 0.5, 1.3, 6, 4, size=14, color=LIGHT_GRAY)
    add_text_box(slide, refs_right, 7, 1.3, 6, 4, size=14, color=LIGHT_GRAY)

    add_text_box(slide, "Thank You!", 1, 5.5, 11.3, 1.2, size=40, color=WHITE, bold=True, align=PP_ALIGN.CENTER)

    # Save
    out_path = os.path.join(OUT_DIR, 'NewsLens_Presentation.pptx')
    prs.save(out_path)
    print(f"[OK] Presentation saved to: {out_path}")
    print(f"     Slides: {len(prs.slides)}")


if __name__ == '__main__':
    main()
