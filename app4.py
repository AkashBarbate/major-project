"""
app.py
------
Streamlit frontend — calls FastAPI backend for all predictions.
Start backend first: uvicorn api:app --port 8000
Then run:           streamlit run app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import re
import os
import sqlite3

# ── Config ──────────────────────────────────────────────────────────────────────
API_BASE    = "http://localhost:8000"
CATEGORIES  = [
    'Politics', 'Science and Technology', 'Business and Economy',
    'International Affairs', 'Education', 'Health', 'Sports', 'Entertainment'
]
OPTUNA_THRESHOLDS = [
    0.6251, 0.7554, 0.4783, 0.7679,
    0.5961, 0.7345, 0.6283, 0.7477,
]

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f8fafc; }
h1 {
    background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    text-align: center;
    padding-bottom: 1rem;
}
.glass {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
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
.conf-text {
    font-size: 11px;
    color: #94a3b8;
    margin-left: 4px;
}
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
    color: white !important;
    font-weight: bold;
    border: none !important;
    border-radius: 8px !important;
}
.stTextArea textarea, .stTextInput input {
    background-color: rgba(15,23,42,0.6) !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}
.api-error {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 8px;
    padding: 12px 16px;
    color: #fca5a5;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ── API helpers ──────────────────────────────────────────────────────────────────

def check_api() -> bool:
    """Check if FastAPI backend is running."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def call_predict(title: str, text: str, url: str = "") -> dict | None:
    """Call /predict endpoint."""
    try:
        r = requests.post(
            f"{API_BASE}/predict",
            json={"title": title, "text": text, "url": url},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()
        st.error(f"API error {r.status_code}: {r.json().get('detail', 'Unknown error')}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Is it running?")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def call_predict_url(url: str) -> dict | None:
    """Call /predict-url endpoint."""
    try:
        r = requests.post(
            f"{API_BASE}/predict-url",
            json={"url": url},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()
        st.error(f"API error {r.status_code}: {r.json().get('detail', 'Unknown error')}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Is it running?")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def call_feedback(text: str, predicted: list, corrected: list) -> dict | None:
    """Call /feedback endpoint."""
    try:
        r = requests.post(
            f"{API_BASE}/feedback",
            json={
                "text":             text[:500],
                "predicted_labels": predicted,
                "corrected_labels": corrected,
            },
            timeout=10,
        )
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def get_feedback_count() -> int:
    try:
        r = requests.get(f"{API_BASE}/feedback/count", timeout=5)
        return r.json().get("count", 0) if r.status_code == 200 else 0
    except Exception:
        return 0


# ── Display helpers ───────────────────────────────────────────────────────────────

def make_radar_chart(label_scores: list) -> go.Figure:
    cats  = [ls['label'] for ls in label_scores]
    vals  = [ls['score'] for ls in label_scores]
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
                range=[0, 1], tickfont_size=9,
                tickcolor='rgba(255,255,255,0.3)',
                gridcolor='rgba(255,255,255,0.1)',
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color='#94a3b8')
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


def confidence_color(score: float) -> str:
    if score >= 0.75: return '#22c55e'
    if score >= 0.50: return '#f59e0b'
    return '#ef4444'


def extractive_summary(text: str, n: int = 3) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    good = [s for s in sentences if len(s.split()) > 8]
    return ' '.join(good[:n]) or text[:300]


def run_xai(response: dict, title: str, content: str):
    """Run XAI locally using already-loaded model (no second load)."""
    predicted_labels = response.get('predicted_labels', [])
    if not predicted_labels:
        return

    try:
        from models.predict import NewsClassifier
        from transformers_interpret import SequenceClassificationExplainer

        # Load classifier — will use cached version from memory
        clf = NewsClassifier()

        label2id = {n: i for i, n in enumerate(CATEGORIES)}
        id2label = {i: n for i, n in enumerate(CATEGORIES)}
        clf.model.config.label2id = label2id
        clf.model.config.id2label = id2label

        explainer = SequenceClassificationExplainer(clf.model, clf.tokenizer)

        # Truncate to prevent index out of bounds
        safe_text = (
            re.sub(r'\s+', ' ', title).strip()[:150]
            + ' '
            + re.sub(r'\s+', ' ', content).strip()[:350]
        )

        tabs = st.tabs(predicted_labels)
        for tab, label in zip(tabs, predicted_labels):
            with tab:
                with st.spinner(f"Generating XAI for {label}..."):
                    try:
                        attrs = dict(explainer(safe_text, class_name=label))
                        filtered = {
                            k: v for k, v in attrs.items()
                            if k not in ['<s>', '</s>', '<pad>', '[SEP]', '[CLS]']
                            and len(k.strip()) > 1
                        }
                        top = sorted(
                            filtered.items(),
                            key=lambda x: abs(x[1]),
                            reverse=True
                        )[:12]
                        if top:
                            df_xai = pd.DataFrame(top, columns=['Keyword', 'Impact'])
                            df_xai.set_index('Keyword', inplace=True)
                            st.bar_chart(df_xai)
                        else:
                            st.info("No significant keywords found.")
                    except Exception as e:
                        st.info(f"XAI unavailable for {label}: {e}")

    except ImportError:
        st.info("Install: pip install transformers-interpret")
    except Exception as e:
        st.info(f"XAI unavailable: {e}")


# ── Main UI ───────────────────────────────────────────────────────────────────────
st.title("📰 AI News Classifier & Credibility Analyser")

# API status indicator
api_ok = check_api()
if api_ok:
    fc = get_feedback_count()
    col_s1, col_s2 = st.columns([1, 4])
    with col_s1:
        st.success("API online")
    with col_s2:
        if fc > 0:
            st.caption(f"📊 {fc} user corrections collected")
else:
    st.markdown("""
    <div class="api-error">
    ⚠️ Backend not running. Start it with:
    <code>uvicorn api:app --host 0.0.0.0 --port 8000</code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Input ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="glass">', unsafe_allow_html=True)
tab_url, tab_text = st.tabs(["🔗 Classify from URL", "✍️ Paste Text"])

response     = None
title_val    = ""
content_val  = ""
url_val      = ""

with tab_url:
    url_val = st.text_input(
        "News article URL",
        placeholder="https://www.ndtv.com/india-news/..."
    )
    if st.button("Fetch & Classify", key="btn_url"):
        if url_val:
            with st.spinner("Scraping and classifying..."):
                response = call_predict_url(url_val)
            if response:
                title_val   = response.get('title', '')
                content_val = url_val   # used for XAI/keywords
        else:
            st.warning("Please enter a URL.")

with tab_text:
    title_val_t   = st.text_input("Headline", placeholder="Article headline...")
    content_val_t = st.text_area("Article content", height=180,
                                  placeholder="Paste full article text here...")
    if st.button("Analyze", key="btn_text"):
        if content_val_t:
            with st.spinner("Classifying..."):
                response = call_predict(
                    title=title_val_t,
                    text=content_val_t,
                )
            title_val   = title_val_t
            content_val = content_val_t
        else:
            st.warning("Please enter article content.")

st.markdown('</div>', unsafe_allow_html=True)

# ── Results ───────────────────────────────────────────────────────────────────────
if response:
    predicted_labels = response.get('predicted_labels', [])
    label_scores     = response.get('label_scores', [])
    confidence       = response.get('overall_confidence', 0)
    is_uncertain     = response.get('is_uncertain', False)
    cred_score       = response.get('credibility_score')
    cred_label_text  = response.get('credibility_label', '')
    cred_signals     = response.get('credibility_signals', [])
    word_count       = response.get('word_count', 0)
    processed_at     = response.get('processed_at', '')

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)

        # Summary
        if content_val and len(content_val) > 100:
            summary = extractive_summary(content_val)
            st.markdown("**Article summary**")
            st.info(summary)

        # Radar chart
        st.markdown("**Confidence scores — all categories**")
        # Sort by category order for consistent radar shape
        ordered_scores = sorted(
            label_scores,
            key=lambda x: CATEGORIES.index(x['label'])
        )
        fig = make_radar_chart(ordered_scores)
        st.plotly_chart(fig, use_container_width=True)

        # Predicted labels with confidence
        st.markdown("**Predicted categories**")
        if is_uncertain:
            st.warning(
                "Low confidence — model is uncertain about this article's category. "
                "It may be outside the training domain."
            )
        else:
            badge_html = ""
            for lbl in predicted_labels:
                score = next(
                    (ls['score'] for ls in label_scores if ls['label'] == lbl), 0
                )
                color = confidence_color(score)
                badge_html += (
                    f'<span class="badge">{lbl}</span>'
                    f'<span class="conf-text" style="color:{color}">'
                    f'{score:.1%}</span> '
                )
            st.markdown(badge_html, unsafe_allow_html=True)

            conf_color = confidence_color(confidence)
            st.caption(
                f"Overall confidence: "
                f"<span style='color:{conf_color};font-weight:600'>"
                f"{confidence:.1%}</span>  ·  "
                f"Word count: {word_count}",
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

        # All scores table
        with st.expander("📊 All category scores"):
            df_scores = pd.DataFrame([
                {
                    'Category':  ls['label'],
                    'Score':     round(ls['score'], 4),
                    'Threshold': round(ls['threshold'], 4),
                    'Predicted': '✓' if ls['predicted'] else '',
                }
                for ls in label_scores
            ])
            st.dataframe(df_scores, use_container_width=True, hide_index=True)

        # XAI
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🧠 XAI — keyword impact")
        st.caption("Words that drove the model's prediction")

        if predicted_labels and not is_uncertain and content_val:
            run_xai(response, title_val, content_val)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Credibility score
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🔍 Credibility Score")

        if cred_score is not None:
            cred_color = (
                '#22c55e' if cred_score >= 75
                else '#f59e0b' if cred_score >= 50
                else '#ef4444'
            )
            st.markdown(f"""
                <div style="text-align:center;padding:16px 0">
                    <div style="font-size:11px;color:#888;
                                text-transform:uppercase;
                                letter-spacing:.08em;margin-bottom:4px">
                        Credibility Score</div>
                    <div style="font-size:56px;font-weight:600;
                                color:{cred_color};line-height:1.1">
                        {cred_score:.0f}</div>
                    <div style="font-size:13px;color:{cred_color};margin-top:4px">
                        {cred_label_text}</div>
                    <div style="font-size:11px;color:#888;margin-top:2px">
                        out of 100</div>
                </div>
            """, unsafe_allow_html=True)

            if cred_signals:
                INVERTED_DISPLAY = {
                    'Subjectivity', 'ALL-CAPS ratio',
                    'Exclamation density', 'Clickbait signals',
                    'Sensational vocabulary', 'Headline questions',
                }
                with st.expander("Signal breakdown"):
                    for sig in cred_signals:
                        st.progress(
                            min(max(float(sig['display_value']), 0.0), 1.0),
                            text=sig['name'],
                        )
        else:
            st.info("Run `python credibility_simple.py` to enable this.")

        st.markdown('</div>', unsafe_allow_html=True)

        # Keywords
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🔑 Article keywords")
        st.caption("What the article is about")

        if content_val and len(content_val) > 50:
            try:
                from keybert import KeyBERT
                @st.cache_resource
                def load_kb():
                    return KeyBERT('distilbert-base-nli-mean-tokens')

                kb  = load_kb()
                kws = kb.extract_keywords(
                    content_val[:2000],
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english', top_n=8,
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
                        f'{kw} <span style="color:{bar_color}">'
                        f'{sc:.2f}</span></span>'
                    )
                st.markdown(kw_html, unsafe_allow_html=True)
            except ImportError:
                st.info("Install: pip install keybert")
            except Exception as e:
                st.info(f"Keywords unavailable: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

        # API response info
        with st.expander("🔧 API response details"):
            st.json({
                "endpoint":     "/predict or /predict-url",
                "model":        response.get("model_version"),
                "processed_at": processed_at,
                "word_count":   word_count,
            })

    # Feedback
    st.divider()
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("✏️ Correct these labels")
    st.caption("Submit corrections to improve future model versions.")

    corrected = st.multiselect(
        "Correct labels:",
        CATEGORIES,
        default=predicted_labels,
        key="correction",
    )
    if st.button("Submit correction", key="btn_feedback"):
        if corrected != predicted_labels:
            result = call_feedback(
                content_val or title_val,
                predicted_labels,
                corrected,
            )
            if result:
                st.success(
                    f"Saved! {result.get('total_corrections', '?')} "
                    f"total corrections."
                )
        else:
            st.info("Labels match prediction — no correction needed.")

    st.markdown('</div>', unsafe_allow_html=True)