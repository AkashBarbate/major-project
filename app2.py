"""
app.py
------
Streamlit app — updated for 8 categories and Optuna thresholds.

Run: streamlit run app.py
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import sqlite3
import pickle
import json
import re
import os
import plotly.graph_objects as go
from transformers import pipeline
from keybert import KeyBERT
from transformers_interpret import SequenceClassificationExplainer
from models.predict import NewsClassifier, CATEGORIES, OPTUNA_THRESHOLDS

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', 'Roboto', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f8fafc; }
h1 {
    background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    text-align: center;
    padding-bottom: 1.5rem;
}
.glass {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}
.badge {
    display: inline-block;
    padding: 5px 14px;
    margin: 4px;
    border-radius: 20px;
    background: linear-gradient(90deg, #6366f1, #a855f7);
    color: white;
    font-weight: 600;
    font-size: 0.88rem;
}
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
    color: white !important;
    font-weight: bold;
    border: none !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ────────────────────────────────────────────────────────────────────────
MODEL_PATH     = 'models/improved_model/best_model'
TOKENIZER_PATH = 'models/improved_model/best_tokenizer'
CRED_MODEL     = 'models/credibility_regressor.pkl'
FEEDBACK_DB    = 'models/feedback.db'

# ── Loaders ───────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_classifier():
    return NewsClassifier(
        model_path     = MODEL_PATH,
        tokenizer_path = TOKENIZER_PATH,
    )

@st.cache_resource
def load_credibility():
    """Load credibility regressor + backbone for CLS extraction."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok   = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    backbone = model.roberta
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    with open(CRED_MODEL, 'rb') as f:
        regressor = pickle.load(f)
    return tok, backbone, regressor

@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=-1, max_length=200, min_length=60,
    )

@st.cache_resource
def load_keybert():
    return KeyBERT('distilbert-base-nli-mean-tokens')

@st.cache_resource
def load_zeroshot():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1,
    )

# ── Helpers ───────────────────────────────────────────────────────────────────────
def scrape_url(url: str) -> tuple[str, str]:
    try:
        from newspaper import Article, Config
        cfg = Config()
        cfg.fetch_images    = False
        cfg.request_timeout = 12
        art = Article(url, config=cfg)
        art.download()
        art.parse()
        if len(art.text.split()) < 80:
            return None, "Article too short or paywalled."
        return art.title or '', art.text
    except Exception as e:
        return None, str(e)


def get_cls_embedding(text: str, tok, backbone) -> np.ndarray:
    enc = tok(
        text, max_length=256, padding='max_length',
        truncation=True, return_tensors='pt'
    )
    with torch.no_grad():
        out = backbone(
            input_ids=enc['input_ids'],
            attention_mask=enc['attention_mask']
        )
    return out.last_hidden_state[0, 0, :].numpy()


def compute_credibility(title, text, url, regressor):
    from language_feature import extract_linguistic_features, get_feature_names
    ling_vec = extract_linguistic_features(title, text, url)
    # Fix: reshape 1D (8,) → 2D (1, 8) for sklearn
    score = float(np.clip(
        regressor.predict(ling_vec.reshape(1, -1))[0], 0, 100
    ))
    return score, ling_vec, get_feature_names()


def get_feedback_count() -> int:
    try:
        conn  = sqlite3.connect(FEEDBACK_DB)
        count = conn.execute(
            "SELECT COUNT(*) FROM corrections"
        ).fetchone()[0]
        conn.close()
        return count
    except:
        return 0


def save_feedback(text: str, predicted: list, corrected: list):
    conn = sqlite3.connect(FEEDBACK_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS corrections
        (id INTEGER PRIMARY KEY, text TEXT,
         predicted TEXT, corrected TEXT, ts TEXT)
    """)
    conn.execute(
        "INSERT INTO corrections VALUES (?,?,?,?,?)",
        (None, text[:1000], str(predicted),
         str(corrected), pd.Timestamp.now().isoformat())
    )
    conn.commit()
    conn.close()


def make_radar_chart(all_scores: dict) -> go.Figure:
    cats  = list(all_scores.keys())
    vals  = list(all_scores.values())
    # Close the polygon
    cats_closed = cats + [cats[0]]
    vals_closed = vals + [vals[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals_closed, theta=cats_closed,
        fill='toself',
        line_color='#7F77DD',
        fillcolor='rgba(127,119,221,0.2)',
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1], tickfont_size=9)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=55, r=55, t=20, b=20),
        showlegend=False,
    )
    return fig


# OOD labels for zero-shot fallback
OOD_LABELS = [
    'Sports', 'Entertainment', 'Crime', 'Travel',
    'Food', 'Fashion', 'Real Estate', *CATEGORIES
]

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("📰 AI News Classifier")

fc = get_feedback_count()
if fc > 0:
    st.caption(f"📊 {fc} user corrections collected for future retraining")

# ── Input ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="glass">', unsafe_allow_html=True)
tab_url, tab_text = st.tabs(["🔗 Classify from URL", "✍️ Paste Text"])

title_val = content_val = url_val = ""
analyze   = False

with tab_url:
    url_val = st.text_input(
        "News article URL",
        placeholder="https://www.ndtv.com/..."
    )
    if st.button("Fetch & Classify"):
        if url_val:
            with st.spinner("Scraping article..."):
                t, c = scrape_url(url_val)
            if t is None:
                st.error(f"Could not scrape: {c}")
            else:
                title_val, content_val = t, c
                st.success(f"Fetched: **{t[:80]}**")
                st.caption(f"{len(c.split())} words extracted")
                analyze = True

with tab_text:
    title_val_t   = st.text_input("Headline", placeholder="Article headline...")
    content_val_t = st.text_area("Article content", height=180,
                                  placeholder="Paste article text here...")
    if st.button("Analyze"):
        if content_val_t:
            title_val, content_val, analyze = title_val_t, content_val_t, True

st.markdown('</div>', unsafe_allow_html=True)

# ── Analysis pipeline ───────────────────────────────────────────────────────────
if analyze and content_val:
    clf = load_classifier()

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)

        # 1. Summarize
        with st.spinner("Summarising article..."):
            try:
                summarizer = load_summarizer()
                summary    = summarizer(content_val[:1024])[0]['summary_text']
                st.markdown("**Summary**")
                st.info(summary)
                classify_text = title_val + ' [SEP] ' + summary
            except:
                classify_text = title_val + ' [SEP] ' + content_val
                summary = None

        # 2. Classify
        with st.spinner("Classifying..."):
            result = clf.predict(
                title=title_val,
                text=content_val,
            )

        # 3. Radar chart
        fig = make_radar_chart(result.all_scores)
        st.plotly_chart(fig, use_container_width=True)

        # 4. Predicted labels
        st.markdown("**Predicted categories**")
        if result.is_uncertain:
            st.warning(
                "Low confidence — checking extended categories..."
            )
            zs     = load_zeroshot()
            zs_res = zs(content_val[:512], candidate_labels=OOD_LABELS)
            st.info(
                f"Best match: **{zs_res['labels'][0]}** "
                f"({zs_res['scores'][0]*100:.0f}%)"
            )
        else:
            badges = " ".join([
                f'<span class="badge">{lbl}</span>'
                for lbl in result.labels
            ])
            st.markdown(badges, unsafe_allow_html=True)
            st.caption(f"Confidence: {result.confidence:.1%}")

        st.markdown('</div>', unsafe_allow_html=True)

        # 5. XAI
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🧠 XAI — keyword impact")
        st.caption("Words that drove the model's prediction "
                   "(different from article keywords below)")

        if result.labels and not result.is_uncertain:
            try:
                from transformers import AutoModelForSequenceClassification as AMSC
                xai_model = AMSC.from_pretrained(MODEL_PATH)
                from transformers import AutoTokenizer as AT
                xai_tok   = AT.from_pretrained(TOKENIZER_PATH)

                # Apply label/id mapping so explainer works
                label2id = {n: i for i, n in enumerate(CATEGORIES)}
                id2label  = {i: n for i, n in enumerate(CATEGORIES)}
                xai_model.config.label2id = label2id
                xai_model.config.id2label = id2label

                explainer = SequenceClassificationExplainer(xai_model, xai_tok)
                tabs      = st.tabs(result.labels)

                for tab, label in zip(tabs, result.labels):
                    with tab:
                        with st.spinner(f"Generating XAI for {label}..."):
                            attrs = dict(explainer(classify_text, class_name=label))
                            filtered = {
                                k: v for k, v in attrs.items()
                                if k not in ['<s>', '</s>', '<pad>', '[SEP]']
                            }
                            top = sorted(
                                filtered.items(),
                                key=lambda x: abs(x[1]), reverse=True
                            )[:15]
                            if top:
                                df_xai = pd.DataFrame(top,
                                    columns=['Keyword', 'Impact'])
                                df_xai.set_index('Keyword', inplace=True)
                                st.bar_chart(df_xai)
            except Exception as e:
                st.info(f"XAI unavailable: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # 6. Credibility score
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🔍 Credibility Score")

        if os.path.exists(CRED_MODEL):
            try:
                with st.spinner("Computing credibility..."):
                    tok_c, backbone, regressor = load_credibility()
                    score, ling_vec, feat_names = compute_credibility(
                        title_val, content_val, url_val,
                        tok_c, backbone, regressor
                    )

                color = (
                    "#22c55e" if score >= 70
                    else "#f59e0b" if score >= 40
                    else "#ef4444"
                )
                label = (
                    "High credibility" if score >= 70
                    else "Moderate credibility" if score >= 40
                    else "Low credibility"
                )

                st.markdown(f"""
                    <div style="text-align:center;padding:16px 0">
                        <div style="font-size:11px;color:#888;
                                    text-transform:uppercase;
                                    letter-spacing:.08em;margin-bottom:4px">
                            Credibility Score</div>
                        <div style="font-size:52px;font-weight:600;
                                    color:{color};line-height:1.1">
                            {score:.0f}</div>
                        <div style="font-size:12px;color:#888">
                            {label} / 100</div>
                    </div>
                """, unsafe_allow_html=True)

                INVERTED = {
                    'Subjectivity', 'ALL-CAPS ratio', 'Exclamation density',
                    'Clickbait signals', 'Sensational vocabulary',
                    'Headline questions',
                }
                with st.expander("Signal breakdown"):
                    for name, val in zip(feat_names, ling_vec):
                        display = float(1 - val) if name in INVERTED else float(val)
                        st.progress(display, text=name)

            except Exception as e:
                st.warning(f"Credibility module unavailable: {e}")
        else:
            st.info("Credibility model not found. "
                    "Run train_credibility.py first.")

        st.markdown('</div>', unsafe_allow_html=True)

        # 7. KeyBERT keywords
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🔑 Article keywords")
        st.caption("What the article is about "
                   "(separate from XAI model explanation)")
        try:
            kb  = load_keybert()
            kws = kb.extract_keywords(
                content_val[:2000],
                keyphrase_ngram_range=(1, 2),
                stop_words='english', top_n=8,
            )
            kw_html = " ".join([
                f'<span style="background:rgba(255,255,255,0.08);'
                f'border:1px solid rgba(255,255,255,0.15);'
                f'padding:3px 10px;border-radius:100px;font-size:12px;">'
                f'{kw} <span style="color:#888">{sc:.2f}</span></span>'
                for kw, sc in kws
            ])
            st.markdown(kw_html, unsafe_allow_html=True)
        except Exception as e:
            st.info(f"KeyBERT unavailable: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # 8. Feedback / active learning
    st.divider()
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("✏️ Correct these labels")
    st.caption("Help improve the model — select the actual correct labels")
    corrected = st.multiselect(
        "Correct labels:",
        CATEGORIES,
        default=result.labels,
        key="correction",
    )
    if st.button("Submit correction"):
        if corrected != result.labels:
            save_feedback(content_val, result.labels, corrected)
            st.success(
                f"Correction saved — thank you! "
                f"({get_feedback_count()} total corrections)"
            )
        else:
            st.info("Labels unchanged — prediction was correct!")
    st.markdown('</div>', unsafe_allow_html=True)