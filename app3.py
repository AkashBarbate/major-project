"""
app.py
------
Multi-label news classifier with credibility scoring, XAI, and keyword extraction.
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
html, body, [class*="css"] {
    font-family: 'Inter', 'Roboto', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #f8fafc;
}
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
    padding: 6px 16px;
    margin: 4px;
    border-radius: 20px;
    background: linear-gradient(90deg, #6366f1, #a855f7);
    color: white;
    font-weight: 600;
    font-size: 0.88rem;
}
.conf-badge {
    display: inline-block;
    padding: 3px 10px;
    margin: 2px 4px;
    border-radius: 12px;
    background: rgba(255,255,255,0.08);
    color: #94a3b8;
    font-size: 0.78rem;
    border: 1px solid rgba(255,255,255,0.1);
}
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
    color: white !important;
    font-weight: bold;
    border: none !important;
    border-radius: 8px !important;
    transition: transform 0.2s !important;
}
.stButton>button:hover {
    transform: translateY(-2px) !important;
}
.stTextArea textarea, .stTextInput input {
    background-color: rgba(15,23,42,0.6) !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ────────────────────────────────────────────────────────────────────────
MODEL_PATH     = 'models/improved_model/best_model'
TOKENIZER_PATH = 'models/improved_model/best_tokenizer'
CRED_MODEL     = 'models/credibility_regressor.pkl'
FEEDBACK_DB    = 'models/feedback.db'

# OOD labels for zero-shot fallback
OOD_LABELS = [
    'Sports', 'Entertainment', 'Crime', 'Travel', 'Food',
    'Fashion', 'Real Estate', *CATEGORIES
]


# ── Resource loaders (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_classifier():
    return NewsClassifier(
        model_path     = MODEL_PATH,
        tokenizer_path = TOKENIZER_PATH,
    )


@st.cache_resource
def load_credibility_module():
    """Load credibility regressor. Returns None if not available."""
    if not os.path.exists(CRED_MODEL):
        return None
    try:
        with open(CRED_MODEL, 'rb') as f:
            regressor = pickle.load(f)
        return regressor
    except Exception as e:
        st.warning(f"Could not load credibility model: {e}")
        return None


@st.cache_resource
def load_summarizer():
    try:
        from transformers import pipeline
        return pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1, max_length=130, min_length=40,
        )
    except Exception:
        return None


@st.cache_resource
def load_keybert():
    try:
        from keybert import KeyBERT
        return KeyBERT('distilbert-base-nli-mean-tokens')
    except Exception:
        return None


@st.cache_resource
def load_zeroshot():
    try:
        from transformers import pipeline
        return pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,
        )
    except Exception:
        return None


# ── Helper functions ─────────────────────────────────────────────────────────────
def scrape_url(url: str) -> tuple[str, str]:
    try:
        from newspaper import Article, Config
        cfg = Config()
        cfg.fetch_images      = False
        cfg.request_timeout   = 12
        cfg.browser_user_agent = (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
        )
        art = Article(url, config=cfg)
        art.download()
        art.parse()
        if len(art.text.split()) < 80:
            return None, "Article too short or paywalled."
        return art.title or '', art.text
    except Exception as e:
        return None, str(e)


def compute_credibility_score(
    title: str, text: str, url: str, regressor
) -> tuple[float, np.ndarray, list[str]]:
    """
    Compute credibility score using linguistic features only.
    Returns (score, feature_vector, feature_names).
    """
    from language_feature import extract_linguistic_features, get_feature_names
    ling_vec = extract_linguistic_features(title, text, url)
    # ── FIX: reshape to (1, -1) for sklearn ──────────────────────────────────
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
    except Exception:
        return 0


def save_feedback(text: str, predicted: list, corrected: list):
    os.makedirs('models', exist_ok=True)
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
    vals  = [round(v, 3) for v in all_scores.values()]
    cats_c = cats + [cats[0]]
    vals_c = vals + [vals[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals_c, theta=cats_c,
        fill='toself',
        line_color='#7F77DD',
        fillcolor='rgba(127,119,221,0.2)',
        hovertemplate='%{theta}<br>Score: %{r:.3f}<extra></extra>',
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[0, 1],
                tickfont_size=9,
                tickcolor='rgba(255,255,255,0.3)',
                gridcolor='rgba(255,255,255,0.1)',
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color='#94a3b8'),
            ),
            bgcolor='rgba(0,0,0,0)',
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=320,
        margin=dict(l=60, r=60, t=20, b=20),
        showlegend=False,
    )
    return fig


def confidence_color(conf: float) -> str:
    """Return hex color based on confidence level."""
    if conf >= 0.75:
        return '#22c55e'   # green
    elif conf >= 0.50:
        return '#f59e0b'   # amber
    else:
        return '#ef4444'   # red


def credibility_label(score: float) -> tuple[str, str]:
    """Return (label, color) for credibility score."""
    if score >= 75:
        return 'High credibility', '#22c55e'
    elif score >= 50:
        return 'Moderate credibility', '#f59e0b'
    else:
        return 'Low credibility', '#ef4444'


# ── App layout ───────────────────────────────────────────────────────────────────
st.title("📰 AI News Classifier & Credibility Analyser")

fc = get_feedback_count()
if fc > 0:
    st.caption(f"📊 {fc} user corrections collected for future retraining")

# ── Input section ────────────────────────────────────────────────────────────────
st.markdown('<div class="glass">', unsafe_allow_html=True)
tab_url, tab_text = st.tabs(["🔗 Classify from URL", "✍️ Paste Text"])

title_val = content_val = url_val = ""
analyze   = False

with tab_url:
    url_val = st.text_input(
        "News article URL",
        placeholder="https://www.ndtv.com/india-news/..."
    )
    if st.button("Fetch & Classify", key="btn_url"):
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
        else:
            st.warning("Please enter a URL first.")

with tab_text:
    title_val_t   = st.text_input(
        "Headline", placeholder="Enter article headline..."
    )
    content_val_t = st.text_area(
        "Article content", height=180,
        placeholder="Paste full article text here..."
    )
    if st.button("Analyze", key="btn_text"):
        if content_val_t:
            title_val, content_val, analyze = title_val_t, content_val_t, True
        else:
            st.warning("Please paste some article content first.")

st.markdown('</div>', unsafe_allow_html=True)

# ── Analysis pipeline ────────────────────────────────────────────────────────────
if analyze and content_val:

    clf        = load_classifier()
    regressor  = load_credibility_module()

    col1, col2 = st.columns([3, 2])

    # ── Left column ──────────────────────────────────────────────────────────
    with col1:

        # 1. Summarize
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        summarizer = load_summarizer()
        if summarizer:
            with st.spinner("Summarising article..."):
                try:
                    summary      = summarizer(
                        content_val[:1024],
                        max_length=130, min_length=40,
                        do_sample=False
                    )[0]['summary_text']
                    st.markdown("**Article summary**")
                    st.info(summary)
                    classify_text = title_val + ' [SEP] ' + summary
                except Exception:
                    classify_text = title_val + ' [SEP] ' + content_val
        else:
            classify_text = title_val + ' [SEP] ' + content_val

        # 2. Classify
        with st.spinner("Classifying with RoBERTa..."):
            result = clf.predict(
                title=title_val,
                text=content_val,
            )

        # 3. Radar chart
        st.markdown("**Confidence scores — all categories**")
        fig = make_radar_chart(result.all_scores)
        st.plotly_chart(fig, use_container_width=True)

        # 4. Predicted labels with per-label confidence
        st.markdown("**Predicted categories**")

        if result.is_uncertain:
            st.warning("Low confidence — checking extended categories...")
            zs = load_zeroshot()
            if zs:
                with st.spinner("Running zero-shot classification..."):
                    try:
                        zs_res = zs(content_val[:512], candidate_labels=OOD_LABELS)
                        st.info(
                            f"Best match: **{zs_res['labels'][0]}** "
                            f"({zs_res['scores'][0]*100:.0f}% confidence)"
                        )
                    except Exception as e:
                        st.info(f"Zero-shot unavailable: {e}")
        else:
            # Show badges with per-label confidence
            badge_html = ""
            for lbl in result.labels:
                score = result.all_scores.get(lbl, 0)
                color = confidence_color(score)
                badge_html += (
                    f'<span class="badge">{lbl}</span>'
                    f'<span class="conf-badge" style="color:{color}">'
                    f'{score:.1%}</span> '
                )
            st.markdown(badge_html, unsafe_allow_html=True)

            # Overall confidence
            conf_color = confidence_color(result.confidence)
            st.caption(
                f"Overall confidence: "
                f"<span style='color:{conf_color};font-weight:600'>"
                f"{result.confidence:.1%}</span>",
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

        # 5. Scores table — all 8 labels
        with st.expander("📊 All label scores"):
            scores_df = pd.DataFrame([
                {
                    'Category': cat,
                    'Score':    round(result.all_scores[cat], 4),
                    'Threshold': round(OPTUNA_THRESHOLDS[i], 4),
                    'Predicted': '✓' if cat in result.labels else '',
                }
                for i, cat in enumerate(CATEGORIES)
            ]).sort_values('Score', ascending=False).reset_index(drop=True)
            st.dataframe(scores_df, use_container_width=True, hide_index=True)

        # 6. XAI section
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🧠 XAI — keyword impact")
        st.caption(
            "Words that drove the model's prediction "
            "(separate from article keywords)"
        )

        if result.labels and not result.is_uncertain:
            try:
                from transformers import (
                    AutoModelForSequenceClassification as AMSC,
                    AutoTokenizer as AT,
                )
                from transformers_interpret import SequenceClassificationExplainer

                xai_model = AMSC.from_pretrained(MODEL_PATH)
                xai_tok   = AT.from_pretrained(TOKENIZER_PATH)

                # Apply label mappings
                label2id = {n: i for i, n in enumerate(CATEGORIES)}
                id2label = {i: n for i, n in enumerate(CATEGORIES)}
                xai_model.config.label2id = label2id
                xai_model.config.id2label = id2label

                explainer = SequenceClassificationExplainer(xai_model, xai_tok)

                # ── FIX: truncate text to prevent index out of bounds ─────────
                # transformers_interpret fails when token count = max_length
                # Use only first 400 chars to stay well within limits
                safe_text = (
                    re.sub(r'\s+', ' ', title_val).strip()[:150]
                    + ' '
                    + re.sub(r'\s+', ' ', content_val).strip()[:400]
                )

                tabs = st.tabs(result.labels)
                for tab, label in zip(tabs, result.labels):
                    with tab:
                        with st.spinner(f"Generating XAI for {label}..."):
                            try:
                                attrs    = dict(explainer(safe_text, class_name=label))
                                filtered = {
                                    k: v for k, v in attrs.items()
                                    if k not in [
                                        '<s>', '</s>', '<pad>',
                                        '[SEP]', '[CLS]', '[PAD]',
                                    ]
                                    and len(k) > 1   # skip single chars
                                }
                                top = sorted(
                                    filtered.items(),
                                    key=lambda x: abs(x[1]),
                                    reverse=True
                                )[:15]

                                if top:
                                    df_xai = pd.DataFrame(
                                        top, columns=['Keyword', 'Impact']
                                    )
                                    df_xai.set_index('Keyword', inplace=True)
                                    st.bar_chart(df_xai)

                                    with st.expander("Raw attribution scores"):
                                        st.json({
                                            k: round(v, 4)
                                            for k, v in top
                                        })
                                else:
                                    st.info("No significant keywords found.")

                            except Exception as e:
                                st.info(f"XAI unavailable for {label}: {e}")

            except ImportError:
                st.info(
                    "Install transformers_interpret: "
                    "pip install transformers-interpret"
                )
            except Exception as e:
                st.info(f"XAI unavailable: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Right column ─────────────────────────────────────────────────────────
    with col2:

        # 7. Credibility score
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🔍 Credibility Score")

        if regressor is not None:
            with st.spinner("Computing credibility..."):
                try:
                    score, ling_vec, feat_names = compute_credibility_score(
                        title_val, content_val, url_val, regressor
                    )
                    cred_label, cred_color = credibility_label(score)

                    st.markdown(f"""
                        <div style="text-align:center;padding:16px 0">
                            <div style="font-size:11px;color:#888;
                                        text-transform:uppercase;
                                        letter-spacing:.08em;
                                        margin-bottom:4px">
                                Credibility Score
                            </div>
                            <div style="font-size:56px;font-weight:600;
                                        color:{cred_color};line-height:1.1">
                                {score:.0f}
                            </div>
                            <div style="font-size:13px;
                                        color:{cred_color};
                                        margin-top:4px">
                                {cred_label}
                            </div>
                            <div style="font-size:11px;color:#888;
                                        margin-top:2px">out of 100</div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Signal breakdown
                    INVERTED_SIGNALS = {
                        'Subjectivity',
                        'ALL-CAPS ratio',
                        'Exclamation density',
                        'Clickbait signals',
                        'Sensational vocabulary',
                        'Headline questions',
                    }
                    with st.expander("Signal breakdown"):
                        for name, val in zip(feat_names, ling_vec):
                            display = float(1 - val) if name in INVERTED_SIGNALS \
                                      else float(val)
                            st.progress(
                                min(max(display, 0.0), 1.0),
                                text=name
                            )

                except Exception as e:
                    st.warning(f"Credibility module error: {e}")
        else:
            st.info(
                "Credibility model not found.\n"
                "Run: `python credibility_simple.py`"
            )

        st.markdown('</div>', unsafe_allow_html=True)

        # 8. KeyBERT keywords
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🔑 Article keywords")
        st.caption(
            "What the article is about "
            "(separate from XAI model explanation)"
        )

        kb = load_keybert()
        if kb:
            try:
                kws = kb.extract_keywords(
                    content_val[:2000],
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    top_n=8,
                )
                kw_html = ""
                for kw, sc in kws:
                    bar_color = (
                        '#22c55e' if sc > 0.6
                        else '#f59e0b' if sc > 0.4
                        else '#94a3b8'
                    )
                    kw_html += (
                        f'<span style="display:inline-block;'
                        f'background:rgba(255,255,255,0.06);'
                        f'border:1px solid rgba(255,255,255,0.12);'
                        f'padding:4px 12px;border-radius:100px;'
                        f'font-size:12px;margin:3px;">'
                        f'{kw} '
                        f'<span style="color:{bar_color}">'
                        f'{sc:.2f}</span></span>'
                    )
                st.markdown(kw_html, unsafe_allow_html=True)
            except Exception as e:
                st.info(f"KeyBERT unavailable: {e}")
        else:
            st.info("Install KeyBERT: `pip install keybert`")

        st.markdown('</div>', unsafe_allow_html=True)

    # 9. Active learning feedback
    st.divider()
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("✏️ Correct these labels")
    st.caption(
        "If the prediction is wrong, select the correct labels. "
        "Your corrections are saved for future model retraining."
    )

    corrected = st.multiselect(
        "Correct labels (modify if needed):",
        CATEGORIES,
        default=result.labels,
        key="correction",
    )
    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        if st.button("Submit correction", key="btn_feedback"):
            if corrected != result.labels:
                save_feedback(content_val, result.labels, corrected)
                st.success(
                    f"Saved! Total corrections: {get_feedback_count()}"
                )
            else:
                st.info("Labels match prediction — no correction needed.")

    st.markdown('</div>', unsafe_allow_html=True)