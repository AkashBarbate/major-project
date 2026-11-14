import streamlit as st
import json
import re
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Import NewsLens 3.0 Core Engine
from src.graph.pipeline import newslens_pipeline
from src.database.session import SessionLocal, init_db
from src.database.models import Article, Claim, EvidenceSnippet, MicroCatalogueEntry, OODReviewItem
from src.web_retrieval.factory import get_web_search_provider
from src.graph.nodes.verify import evaluate_stance, aggregate_verdict
from src.graph.nodes.claims import normalize_query
from src.graph.nodes.classify import CATEGORIES, OPTUNA_THRESHOLDS
from src.taxonomy.llm_suggester import suggest_micro_label_with_llm
from src.web_retrieval.scraper import extract_article_content
from src.graph.nodes.micro_retrieval import register_micro_label_in_vector_db

# Page configuration
st.set_page_config(
    page_title="NewsLens 3.0 — Evidence-Aware News Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Dark Theme & Typography */
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #111827 100%);
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Gradient */
    .main-title {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.4rem;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Cards & Containers */
    .lens-card {
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 18px;
        backdrop-filter: blur(8px);
    }
    
    /* Badges */
    .macro-badge {
        display: inline-block;
        padding: 6px 14px;
        margin: 4px;
        border-radius: 20px;
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        color: white;
        font-weight: 600;
        font-size: 0.88rem;
        box-shadow: 0 2px 6px rgba(59, 130, 246, 0.25);
    }
    .micro-badge {
        display: inline-block;
        padding: 5px 12px;
        margin: 3px;
        border-radius: 16px;
        background: rgba(99, 102, 241, 0.18);
        border: 1px solid rgba(129, 140, 248, 0.4);
        color: #a5b4fc;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .ood-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .ood-clean {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .ood-alert {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Verdict Badges */
    .verdict-supported {
        background: #14532d;
        color: #86efac;
        border: 1px solid #22c55e;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .verdict-refuted {
        background: #7f1d1d;
        color: #fca5a5;
        border: 1px solid #ef4444;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .verdict-mixed {
        background: #78350f;
        color: #fde047;
        border: 1px solid #f59e0b;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .verdict-nei {
        background: #334155;
        color: #cbd5e1;
        border: 1px solid #64748b;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    
    .stance-supports { color: #4ade80; font-weight: 600; }
    .stance-refutes  { color: #f87171; font-weight: 600; }
    .stance-neutral  { color: #94a3b8; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# Suppress noisy tokenizer & transformer logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

# Initialize DB
init_db()


@st.cache_resource(show_spinner="⚡ Preloading and warming up AI models into memory for fast prediction...")
def warmup_system_models():
    """
    Loads all deep learning models once into RAM at application launch.
    Eliminates cold-start delays during interactive prediction.
    """
    from src.graph.nodes.classify import get_classifier
    from src.graph.nodes.micro_retrieval import get_embedder, load_catalogue
    from src.evidence_kb.local_kb import get_local_evidence_kb
    from src.graph.nodes.verify import get_nli_scorer

    get_classifier()
    get_embedder()
    load_catalogue()
    get_local_evidence_kb()
    get_nli_scorer()
    return True


# Preload and warm up models once
warmup_system_models()

# Sidebar: System Governance & Configuration
with st.sidebar:
    st.markdown("### 🔍 NewsLens 3.0")
    st.caption("Evidence-Aware Decision-Support System")
    st.success("⚡ **Models Warmed & Ready**")
    st.markdown("---")
    
    search_pref = st.selectbox(
        "Live Search Provider",
        options=["auto (Tavily/DDG)", "duckduckgo", "tavily"],
        help="Selects search engine for claim verification."
    )

    st.markdown("#### ⚡ Speed & Verification")
    verification_mode = st.radio(
        "Verification Mode",
        ["Fast (Top 2 Claims)", "Thorough (Top 3 Claims)"],
        index=0,
        help="Fast mode verifies the top 2 check-worthy claims in parallel for sub-5 second response time."
    )
    max_claims_setting = 2 if "Fast" in verification_mode else 3
    
    st.markdown("#### Epistemic Policy")
    st.info(
        "**Decision Support Only**: NewsLens does not determine absolute truth. "
        "Verdicts reflect retrieved, time-stamped evidence from configured sources. "
        "**NEI (Not Enough Information)** means evidence is inconclusive, not false."
    )
    
    st.markdown("---")
    st.markdown("#### System Components")
    st.write("• **Macro Classifier**: RoBERTa (8 categories)")
    st.write("• **Calibration**: Temperature Scaling ($T=1.2875$)")
    st.write("• **Micro Taxonomy**: 45+ Active Semantic Prototypes")
    st.write("• **State Orchestrator**: LangGraph Engine")
    st.write("• **Persistence**: SQLite (`data/newslens.db`)")


# Tabs Layout
tab_analysis, tab_sandbox, tab_taxonomy, tab_overview = st.tabs([
    "📰 Article Analysis & Verification",
    "🧪 Claim Verification Sandbox",
    "🗂️ Taxonomy & OOD Review Queue",
    "⚙️ System Architecture & Metrics"
])


# ==============================================================================
# TAB 1: ARTICLE ANALYSIS & DECISION SUPPORT
# ==============================================================================
with tab_analysis:
    st.markdown('<div class="main-title">NewsLens 3.0</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Hierarchical Classification • Claim Extraction • Hybrid Evidence Verification</div>', unsafe_allow_html=True)
    
    # State initialization for article persistence across Streamlit reruns
    if "scraped_headline" not in st.session_state:
        st.session_state["scraped_headline"] = ""
    if "scraped_body" not in st.session_state:
        st.session_state["scraped_body"] = ""
    if "scraped_title_edit" not in st.session_state:
        st.session_state["scraped_title_edit"] = ""
    if "scraped_body_edit" not in st.session_state:
        st.session_state["scraped_body_edit"] = ""
    if "scraped_url" not in st.session_state:
        st.session_state["scraped_url"] = ""
    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None
    if "analysis_title" not in st.session_state:
        st.session_state["analysis_title"] = ""

    with st.expander("Input Options & Article Submission", expanded=True):
        input_mode = st.radio("Choose Input Method:", ["Raw Text", "Fetch from News URL"], horizontal=True)
        
        headline_input = ""
        body_input = ""
        url_input = ""
        
        if input_mode == "Raw Text":
            headline_input = st.text_input("Article Headline / Title", placeholder="e.g. RBI raises benchmark repo rate by 25 basis points to 6.75%")
            body_input = st.text_area("Article Body Text", height=180, placeholder="Paste clean article content here...")
        else:
            url_input = st.text_input(
                "Article Full URL",
                value=st.session_state.get("scraped_url", ""),
                placeholder="https://timesofindia.indiatimes.com/..."
            )
            st.session_state["scraped_url"] = url_input

            col_s1, col_s2 = st.columns([2, 1])
            with col_s1:
                scrape_btn = st.button("📥 Scrape Article Content")
            with col_s2:
                if st.session_state.get("scraped_body") or st.session_state.get("scraped_body_edit"):
                    if st.button("🗑️ Clear Scraped Content"):
                        st.session_state["scraped_headline"] = ""
                        st.session_state["scraped_body"] = ""
                        st.session_state["scraped_title_edit"] = ""
                        st.session_state["scraped_body_edit"] = ""
                        st.rerun()

            if scrape_btn:
                if url_input and url_input.strip():
                    with st.spinner("Scraping news article from URL..."):
                        t, b = extract_article_content(url_input.strip())
                        if b and len(b.strip()) >= 25:
                            st.session_state["scraped_headline"] = t
                            st.session_state["scraped_body"] = b
                            st.session_state["scraped_title_edit"] = t
                            st.session_state["scraped_body_edit"] = b
                            st.success(f"Scraped {len(b.split())} words ({len(b.strip())} characters) successfully! Ready for analysis.")
                            st.rerun()
                        else:
                            st.error("Could not extract article text from this URL. The URL may be invalid, truncated (missing article ID), paywalled, or blocking automated requests. Please check the URL or paste the article text under 'Raw Text'.")
                else:
                    st.warning("Please provide a valid URL.")

            # Display scraped content fields for review and editing
            saved_body = st.session_state.get("scraped_body_edit") or st.session_state.get("scraped_body", "")
            saved_title = st.session_state.get("scraped_title_edit") or st.session_state.get("scraped_headline", "")
            
            if saved_body:
                st.markdown("##### 📄 Scraped Content Preview (Editable)")
                headline_input = st.text_input(
                    "Article Headline / Title",
                    value=saved_title,
                    key="scraped_title_edit"
                )
                body_input = st.text_area(
                    "Article Body Text",
                    value=saved_body,
                    height=180,
                    key="scraped_body_edit"
                )
                # Keep state synced with manual edits
                st.session_state["scraped_headline"] = headline_input
                st.session_state["scraped_body"] = body_input
            else:
                headline_input = ""
                body_input = ""

        run_btn = st.button("⚡ Run Full NewsLens 3.0 Analysis", type="primary")

    if run_btn:
        if input_mode == "Fetch from News URL":
            # Priority: active text from edited text area or persisted session state
            active_body = (
                st.session_state.get("scraped_body_edit", "").strip()
                or st.session_state.get("scraped_body", "").strip()
                or body_input.strip()
            )
            active_headline = (
                st.session_state.get("scraped_title_edit", "").strip()
                or st.session_state.get("scraped_headline", "").strip()
                or headline_input.strip()
            )

            # Auto-scrape fallback if in URL mode and user didn't hit Scrape first
            if not active_body and url_input and url_input.strip():
                with st.spinner("Auto-fetching article content from URL..."):
                    t, b = extract_article_content(url_input.strip())
                    if b and len(b.strip()) >= 25:
                        active_headline = t
                        active_body = b
                        st.session_state["scraped_headline"] = t
                        st.session_state["scraped_body"] = b
                        st.session_state["scraped_title_edit"] = t
                        st.session_state["scraped_body_edit"] = b
                    else:
                        st.error("Could not auto-scrape article text. The URL may be invalid, truncated (missing article ID), or blocking automated requests. Please check the URL or paste the article text under 'Raw Text'.")

            headline_input = active_headline
            body_input = active_body

        if not body_input or len(body_input.strip()) < 25:
            st.warning("Please provide sufficient article text (minimum 25 characters). If using URL mode, click 'Scrape Article Content' or paste the text under 'Raw Text'.")
        else:
            with st.spinner("Executing LangGraph pipeline (Ingestion ➔ Classification ➔ Micro-retrieval ➔ Credibility ➔ Claim Extraction ➔ Live Verification ➔ SQLite)..."):
                initial_state = {
                    "raw_title": headline_input,
                    "raw_text": body_input,
                    "url": url_input or None,
                    "max_claims": max_claims_setting,
                }
                
                # Execute LangGraph Pipeline
                exec_result = newslens_pipeline.invoke(initial_state)
                st.session_state["analysis_result"] = exec_result
                st.session_state["analysis_title"] = headline_input or exec_result.get("clean_title", "Analyzed Article")

    # Render Persisted Analysis Results
    result = st.session_state.get("analysis_result")
    if result is not None:
        st.markdown("---")
        res_header_col1, res_header_col2 = st.columns([4, 1])
        with res_header_col1:
            st.success(f"Analysis complete for: **{st.session_state.get('analysis_title', 'Article')}**")
        with res_header_col2:
            if st.button("🔄 Clear Analysis"):
                st.session_state["analysis_result"] = None
                st.session_state["analysis_title"] = ""
                st.rerun()

        # ------------------------------------------------------------------
        # SECTION A: CATEGORIZATION & CREDIBILITY OVERVIEW
        # ------------------------------------------------------------------
        st.markdown("### 1. Classification & Behavioral Reliability")
        col1, col2, col3 = st.columns([1.2, 1.2, 1.4])

        with col1:
            st.markdown('<div class="lens-card">', unsafe_allow_html=True)
            st.markdown("#### Macro Categories (RoBERTa)")
            predicted_macros = [m for m in result.get("macro_predictions", []) if m["predicted"]]
            
            if predicted_macros:
                for m in predicted_macros:
                    st.markdown(f'<span class="macro-badge">{m["label"]} • {m["score"]*100:.1f}%</span>', unsafe_allow_html=True)
            else:
                st.write("*No category exceeded Optuna threshold.*")
            
            st.caption(f"Overall Confidence: **{result.get('macro_confidence', 0.5)*100:.1f}%** | Calibrated ($T=1.2875$)")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="lens-card">', unsafe_allow_html=True)
            st.markdown("#### Retrieved Micro-Labels")
            micro_labels = result.get("micro_labels", [])
            has_llm_discovered = False
            if micro_labels:
                for mic in micro_labels:
                    if mic.get("is_llm_discovered"):
                        has_llm_discovered = True
                        st.markdown(
                            f'<span class="micro-badge" style="border: 1px solid #818cf8; background: rgba(99, 102, 241, 0.35); color: #e0e7ff;">'
                            f'✨ {mic["name"]} ({mic["macro_category"]})</span>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(f'<span class="micro-badge">{mic["name"]} ({mic["macro_category"]})</span>', unsafe_allow_html=True)

                if has_llm_discovered:
                    st.caption("✨ *Novel topic(s) discovered via LLM and indexed into vector DB for future searches.*")
            else:
                st.write("*No specific micro-label matched catalogue.*")

            ood_score = result.get("ood_score", 0.0)
            review_req = result.get("ood_review_required", False)
            if review_req:
                st.markdown(f'<br><span class="ood-pill ood-alert">⚠️ Novel Topic Candidate (OOD: {ood_score:.2f})</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<br><span class="ood-pill ood-clean">✓ In-Taxonomy Prototype (OOD: {ood_score:.2f})</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="lens-card">', unsafe_allow_html=True)
            st.markdown("#### Credibility & Manipulation Cues")
            cred_score = result.get("credibility_score")
            cred_lbl = result.get("credibility_label") or "Analyzed"
            if cred_score is not None:
                st.metric(label="Linguistic Reliability Score", value=f"{cred_score:.1f} / 100", delta=cred_lbl)
            
            with st.expander("Inspect 8 Transparent Behavioral Signals"):
                for sig in result.get("credibility_signals", []):
                    st.write(f"• **{sig['name']}**: `{sig['value']:.3f}` (Clean display: `{sig['display_value']:.2f}`)")
            st.markdown('</div>', unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # SECTION B: FACTUAL CLAIM EXTRACTION & EVIDENCE VERIFICATION
        # ------------------------------------------------------------------
        st.markdown("### 2. Evidence-Aware Claim Verification")
        claims = result.get("claims", [])
        div_summary = result.get("source_diversity_summary", {})

        st.caption(
            f"Extracted **{len(claims)} check-worthy factual claims** • "
            f"Retrieved **{div_summary.get('total_evidence_retrieved', 0)} evidence passages** from "
            f"**{div_summary.get('unique_source_groups', 0)} independent source groups**."
        )

        if not claims:
            st.info("No check-worthy factual assertions detected in this article.")
        else:
            for idx, clm in enumerate(claims, 1):
                verdict = clm.get("verdict", "NEI")
                v_class = f"verdict-{verdict.lower()}"
                
                with st.container():
                    st.markdown(f"""
                    <div class="lens-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <span style="font-weight: 700; font-size: 1.05rem; color: #e2e8f0;">Claim {idx}</span>
                            <span class="{v_class}">{verdict} (Conf: {clm.get('confidence', 0.5)*100:.0f}%)</span>
                        </div>
                        <blockquote style="border-left: 3px solid #38bdf8; padding-left: 14px; margin: 0 0 14px 0; color: #f8fafc; font-size: 1.05rem;">
                            "{clm['text']}"
                        </blockquote>
                        <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 12px;">
                            Check-worthiness score: <b>{clm.get('checkworthiness_score', 0):.2f}</b> | Normalized query: <i>{clm.get('normalized_query')}</i>
                        </div>
                    """, unsafe_allow_html=True)

                    evidence_list = clm.get("evidence", [])
                    if not evidence_list:
                        st.write("*No direct corroborating or refuting passages found in configured evidence sources (Status: NEI).*")
                    else:
                        st.markdown("##### Retrieved Evidence Passages & Stance:")
                        for ev in evidence_list:
                            stance = ev.get("stance", "neutral")
                            s_class = f"stance-{stance.lower()}"
                            
                            st.markdown(f"""
                            <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px; margin-bottom: 8px; border: 1px solid rgba(255,255,255,0.04);">
                                <div style="display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 4px;">
                                    <span><b>{ev.get('source_group', 'Source')}</b> ({ev.get('domain')}) • <i>{ev.get('source_type', 'reporting')}</i></span>
                                    <span class="{s_class}">{stance.upper()} (NLI: {ev.get('nli_score', 0.5):.2f})</span>
                                </div>
                                <div style="font-size: 0.92rem; color: #cbd5e1; margin-bottom: 6px;">
                                    "{ev.get('passage')}"
                                </div>
                                <a href="{ev.get('url')}" target="_blank" style="color: #38bdf8; font-size: 0.8rem; text-decoration: none;">🔗 View Source Link</a>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # SECTION C: LIMITATIONS & AUDIT NOTICE
        # ------------------------------------------------------------------
        with st.expander("🛡️ Epistemic Limits & Methodological Notice", expanded=False):
            st.markdown("""
            - **Decision Support**: NewsLens 3.0 provides traceable evidence rather than declaring absolute truth.
            - **NEI Policy**: A verdict of *Not Enough Information (NEI)* means configured search sources did not provide decisive evidence at analysis time. It is **not** a marker of falsehood.
            - **Source Diversity**: Syndicated wire reports across multiple domains are grouped into their parent organizations to prevent false multi-confirmation.
            - **Human Approval**: Novel topic proposals remain in an approval queue and are not autonomously added to the taxonomy.
            """)


# ==============================================================================
# TAB 2: CLAIM VERIFICATION SANDBOX
# ==============================================================================
with tab_sandbox:
    st.markdown("### 🧪 Standalone Claim Verification Sandbox")
    st.caption("Verify isolated factual assertions with real-time web retrieval and NLI stance aggregation.")

    claim_input = st.text_input("Enter Claim to Verify", placeholder="e.g. WHO declared the end of the global COVID-19 emergency in May 2023")
    col_sb1, col_sb2 = st.columns([1, 1])
    with col_sb1:
        provider_sb = st.selectbox("Search Provider", ["auto", "duckduckgo", "tavily"], index=0)
    with col_sb2:
        max_res = st.slider("Max Passages to Retrieve", min_value=2, max_value=8, value=4)

    if st.button("🔎 Verify Factual Claim", type="primary"):
        if not claim_input or len(claim_input.strip()) < 10:
            st.warning("Please provide a factual claim statement.")
        else:
            with st.spinner("Searching and scoring evidence..."):
                provider = get_web_search_provider(preferred=provider_sb)
                norm_q = normalize_query(claim_input)
                raw_ev = provider.search(query=norm_q, max_results=max_res)

                ev_records = []
                for ev in raw_ev:
                    st_val, nli_c = evaluate_stance(claim_input, ev.passage)
                    ev_records.append({
                        "url": ev.url,
                        "title": ev.title,
                        "domain": ev.domain,
                        "source_group": ev.source_group,
                        "source_type": ev.source_type,
                        "passage": ev.passage,
                        "stance": st_val,
                        "nli_score": nli_c,
                        "retrieved_at": ev.retrieved_at,
                    })

                v_val, v_conf, v_div = aggregate_verdict(ev_records)

            st.markdown(f"#### Aggregated Verdict: **{v_val.upper()}** (Confidence: {v_conf*100:.0f}%)")
            st.caption(f"Sources analyzed: {v_div.get('total_sources', 0)} | Unique groups: {v_div.get('unique_groups', 0)}")

            for ev in ev_records:
                st.markdown(f"""
                <div class="lens-card" style="padding: 14px; margin-bottom: 8px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.85rem;">
                        <span><b>{ev['source_group']}</b> ({ev['domain']})</span>
                        <span class="stance-{ev['stance']}">{ev['stance'].upper()} ({ev['nli_score']:.2f})</span>
                    </div>
                    <p style="margin: 8px 0; font-size: 0.93rem;">"{ev['passage']}"</p>
                    <a href="{ev['url']}" target="_blank" style="color: #38bdf8; font-size:0.82rem;">🔗 Source URL</a>
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# TAB 3: TAXONOMY & OOD REVIEW QUEUE
# ==============================================================================
with tab_taxonomy:
    st.markdown("### 🗂️ Human-in-the-Loop Taxonomy Governance")
    st.caption("Review Out-of-Distribution (OOD) novel topic candidates and inspect the approved micro-label catalogue.")

    db = SessionLocal()
    pending_items = db.query(OODReviewItem).filter_by(status="pending").all()
    
    st.markdown(f"#### Pending Novel Topic Candidates ({len(pending_items)})")
    if not pending_items:
        st.info("No candidates pending review. When an article's embedding distance exceeds threshold, it appears here.")
    else:
        for it in pending_items:
            with st.container():
                st.markdown(f"""
                <div class="lens-card">
                    <div style="font-size: 1.05rem; font-weight: 600; color: #e2e8f0;">Headline: {it.headline}</div>
                    <div style="color: #94a3b8; font-size: 0.88rem; margin: 6px 0;">
                        OOD Score: <b>{it.ood_score:.3f}</b> | Nearest Prototype: <b>{it.nearest_micro_label}</b> (Similarity: {it.similarity:.3f})
                    </div>
                """, unsafe_allow_html=True)
                
                # Check if LLM suggestion exists in session state
                llm_key = f"llm_suggest_{it.id}"
                sugg = st.session_state.get(llm_key, {})
                default_name = sugg.get("canonical_name") or it.nearest_micro_label or ""
                default_macro = sugg.get("macro_category") or "Politics"
                macro_idx = CATEGORIES.index(default_macro) if default_macro in CATEGORIES else 0

                c_act1, c_act2, c_act3, c_act4 = st.columns([1.5, 1.2, 1, 0.8])
                with c_act1:
                    new_label_name = st.text_input("Canonical Micro-Label Name", value=default_name, key=f"name_{it.id}")
                with c_act2:
                    assigned_macro = st.selectbox("Assigned Macro Category", CATEGORIES, index=macro_idx, key=f"cat_{it.id}")
                with c_act3:
                    st.write("")
                    st.write("")
                    if st.button("✨ Ask LLM", key=f"btn_llm_{it.id}", help="Use LangChain Gemini to suggest micro-label"):
                        with st.spinner("Consulting Gemini..."):
                            rec = suggest_micro_label_with_llm(it.headline, nearest_category=it.nearest_micro_label or "")
                            st.session_state[llm_key] = rec
                            st.rerun()
                with c_act4:
                    st.write("")
                    st.write("")
                    if st.button("✓ Approve", key=f"app_{it.id}", type="primary"):
                        clean_name = new_label_name.strip().lower()
                        if not clean_name:
                            st.warning("Please provide a valid micro-label name.")
                        else:
                            try:
                                existing = db.query(MicroCatalogueEntry).filter_by(name=clean_name).first()
                                if existing:
                                    # Label already exists in catalogue -> merge seed examples safely without IntegrityError
                                    seeds = list(existing.seed_examples or [])
                                    if it.headline not in seeds:
                                        seeds.append(it.headline)
                                        existing.seed_examples = seeds
                                    it.status = "approved"
                                    it.reviewed_by = "Analyst"
                                    it.proposed_name = existing.name
                                    db.commit()
                                    st.success(f"Merged candidate into existing label '{existing.name}'!")
                                else:
                                    # Genuinely novel micro-label -> insert and index in vector DB
                                    definition_text = sugg.get("definition") or f"Approved cluster from: {it.headline}"
                                    aliases_list = sugg.get("aliases") or [clean_name]
                                    register_micro_label_in_vector_db(
                                        name=clean_name,
                                        macro_category=assigned_macro,
                                        definition=definition_text,
                                        aliases=aliases_list,
                                        seed_example=it.headline,
                                    )
                                    it.status = "approved"
                                    it.reviewed_by = "Analyst"
                                    it.proposed_name = clean_name
                                    db.commit()
                                    st.success(f"Approved novel micro-label '{clean_name}' and indexed into vector DB!")
                                st.rerun()
                            except Exception as e:
                                db.rollback()
                                st.error(f"Approval failed: {e}")

                if sugg:
                    st.caption(f"💡 **LLM Proposal**: Definition: *{sugg.get('definition')}* | Aliases: {', '.join(sugg.get('aliases', []))}")

                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Approved Micro-Label Catalogue")
    catalogue_entries = db.query(MicroCatalogueEntry).all()
    db.close()

    cat_df = pd.DataFrame([
        {
            "ID": e.id,
            "Micro-Label": e.name,
            "Macro Category": e.macro_category,
            "Status": e.status,
            "Definition": e.definition[:90] + "...",
        }
        for e in catalogue_entries
    ])
    st.dataframe(cat_df)


# ==============================================================================
# TAB 4: SYSTEM ARCHITECTURE & METRICS
# ==============================================================================
with tab_overview:
    st.markdown("### ⚙️ System Metrics & Database Status")
    
    db = SessionLocal()
    total_articles = db.query(Article).count()
    total_claims = db.query(Claim).count()
    total_snippets = db.query(EvidenceSnippet).count()
    db.close()

    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    c_m1.metric("Persisted Articles", total_articles)
    c_m2.metric("Extracted Claims", total_claims)
    c_m3.metric("Evidence Snippets", total_snippets)
    c_m4.metric("Approved Micro-Labels", len(catalogue_entries))

    st.markdown("#### Frozen Macro Taxonomy & Validation Thresholds")
    thresh_df = pd.DataFrame({
        "Macro Category": CATEGORIES,
        "Optuna Tuned Threshold": [round(float(t), 4) for t in OPTUNA_THRESHOLDS],
        "Calibrated Temp": 1.2875,
    })
    st.table(thresh_df)

    st.markdown("#### Fast API Backend Endpoints")
    st.code("""
# Primary Endpoints:
POST /articles        -> Full LangGraph pipeline execution
POST /articles/url    -> Scrape news URL and analyze
GET  /articles        -> List analyzed records from SQLite
POST /classify        -> Fast macro + micro classification
POST /claims/extract  -> Check-worthiness scoring & candidate extraction
POST /verify          -> Standalone claim verification against live web
GET  /taxonomy        -> Micro-label catalogue
GET  /taxonomy/ood-queue -> Pending novel topic candidates
POST /annotations     -> Inter-annotator agreement submissions
    """, language="python")
