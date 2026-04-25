"""
api.py
------
FastAPI backend for the News Classifier.
Exposes REST endpoints that app.py calls.

Run: uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
  POST /predict          - classify article text
  POST /predict-url      - scrape + classify from URL
  GET  /health           - health check
  GET  /categories       - list all 8 categories
  GET  /feedback/count   - number of corrections collected
  POST /feedback         - submit a label correction
"""

import os
import re
import sqlite3
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from models.predict import NewsClassifier, CATEGORIES, OPTUNA_THRESHOLDS
from language_feature import extract_linguistic_features, get_feature_names

# ── Config ──────────────────────────────────────────────────────────────────────
MODEL_PATH     = 'models/improved_model/best_model'
TOKENIZER_PATH = 'models/improved_model/best_tokenizer'
CRED_MODEL     = 'models/credibility_regressor.pkl'
FEEDBACK_DB    = 'models/feedback.db'

# ── Global model store ───────────────────────────────────────────────────────────
# Loaded once at startup, shared across all requests
_clf        = None
_regressor  = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    global _clf, _regressor

    print("Loading classifier...")
    _clf = NewsClassifier(
        model_path     = MODEL_PATH,
        tokenizer_path = TOKENIZER_PATH,
    )

    if os.path.exists(CRED_MODEL):
        print("Loading credibility regressor...")
        with open(CRED_MODEL, 'rb') as f:
            _regressor = pickle.load(f)
        print("Credibility regressor loaded.")
    else:
        print("Credibility model not found — scoring unavailable.")

    print("API ready.")
    yield

    # Cleanup
    del _clf
    del _regressor
    print("Models unloaded.")


# ── FastAPI app ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title        = "News Classifier API",
    description  = "Multi-label news classification with credibility scoring",
    lifespan     = lifespan,
)

# Allow Streamlit (port 8501) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Pydantic models ──────────────────────────────────────────────────────────────

class ArticleRequest(BaseModel):
    title:   str = Field(default="", description="Article headline")
    text:    str = Field(...,        description="Article body text")
    url:     str = Field(default="", description="Source URL (optional)")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "RBI raises repo rate to control inflation",
                "text":  "The Reserve Bank of India on Friday raised its "
                         "benchmark lending rate by 25 basis points...",
                "url":   "https://www.thehindu.com/business/economy/rbi-rate-hike",
            }
        }


class URLRequest(BaseModel):
    url: str = Field(..., description="Full URL of news article to scrape and classify")

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://www.bbc.com/news/world-asia-india-12345678"
            }
        }


class FeedbackRequest(BaseModel):
    text:            str       = Field(..., description="Article text")
    predicted_labels: list[str] = Field(..., description="Labels predicted by model")
    corrected_labels: list[str] = Field(..., description="Correct labels from user")


class LabelScore(BaseModel):
    label:     str
    score:     float
    threshold: float
    predicted: bool


class CredibilitySignal(BaseModel):
    name:         str
    value:        float
    display_value: float   # inverted for negative signals


class PredictionResponse(BaseModel):
    # Classification
    predicted_labels:  list[str]
    label_scores:      list[LabelScore]
    overall_confidence: float
    is_uncertain:      bool

    # Credibility
    credibility_score:   Optional[float]
    credibility_label:   Optional[str]
    credibility_signals: Optional[list[CredibilitySignal]]

    # Article info
    title:      str
    word_count: int
    url:        str

    # Meta
    model_version: str = "roberta-8cat-v1"
    processed_at:  str


class HealthResponse(BaseModel):
    status:               str
    classifier_loaded:    bool
    credibility_loaded:   bool
    categories:           list[str]


# ── Helper functions ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def compute_credibility(
    title: str, text: str, url: str
) -> tuple[float, str, list[CredibilitySignal]]:
    """Compute credibility score and return structured signals."""
    if _regressor is None:
        return None, None, None

    ling_vec  = extract_linguistic_features(title, text, url)
    score     = float(np.clip(
        _regressor.predict(ling_vec.reshape(1, -1))[0], 0, 100
    ))

    if score >= 75:
        label = "High credibility"
    elif score >= 50:
        label = "Moderate credibility"
    else:
        label = "Low credibility"

    INVERTED = {
        'Subjectivity', 'ALL-CAPS ratio', 'Exclamation density',
        'Clickbait signals', 'Sensational vocabulary', 'Headline questions',
    }

    signals = [
        CredibilitySignal(
            name          = name,
            value         = float(val),
            display_value = float(1 - val) if name in INVERTED else float(val),
        )
        for name, val in zip(get_feature_names(), ling_vec)
    ]

    return score, label, signals


def build_prediction_response(
    title: str, text: str, url: str
) -> PredictionResponse:
    """Core prediction logic shared by /predict and /predict-url."""
    if _clf is None:
        raise HTTPException(status_code=503, detail="Classifier not loaded")

    # Classify
    result = _clf.predict(title=title, text=text)

    # Build label scores list
    label_scores = [
        LabelScore(
            label     = cat,
            score     = round(result.all_scores[cat], 4),
            threshold = round(OPTUNA_THRESHOLDS[i], 4),
            predicted = cat in result.labels,
        )
        for i, cat in enumerate(CATEGORIES)
    ]

    # Sort by score descending
    label_scores.sort(key=lambda x: x.score, reverse=True)

    # Credibility
    cred_score, cred_label, cred_signals = compute_credibility(
        clean_text(title), clean_text(text), url
    )

    return PredictionResponse(
        predicted_labels    = result.labels,
        label_scores        = label_scores,
        overall_confidence  = round(result.confidence, 4),
        is_uncertain        = result.is_uncertain,
        credibility_score   = round(cred_score, 1) if cred_score else None,
        credibility_label   = cred_label,
        credibility_signals = cred_signals,
        title               = title[:200],
        word_count          = len(text.split()),
        url                 = url,
        processed_at        = datetime.now().isoformat(),
    )


def get_db_connection():
    os.makedirs('models', exist_ok=True)
    conn = sqlite3.connect(FEEDBACK_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS corrections (
            id        INTEGER PRIMARY KEY,
            text      TEXT,
            predicted TEXT,
            corrected TEXT,
            ts        TEXT
        )
    """)
    return conn


# ── Endpoints ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check if the API and models are ready."""
    return HealthResponse(
        status             = "ok",
        classifier_loaded  = _clf is not None,
        credibility_loaded = _regressor is not None,
        categories         = CATEGORIES,
    )


@app.get("/categories", tags=["System"])
def get_categories():
    """Return all supported classification categories with their thresholds."""
    return {
        "categories": [
            {
                "name":      cat,
                "threshold": round(OPTUNA_THRESHOLDS[i], 4),
            }
            for i, cat in enumerate(CATEGORIES)
        ]
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Classification"])
def predict_text(request: ArticleRequest):
    """
    Classify a news article from raw text.

    Returns predicted labels, confidence scores for all 8 categories,
    and credibility analysis.
    """
    if not request.text or len(request.text.strip()) < 20:
        raise HTTPException(
            status_code=422,
            detail="Article text too short. Minimum 20 characters."
        )

    return build_prediction_response(
        title = clean_text(request.title),
        text  = clean_text(request.text),
        url   = request.url,
    )


@app.post("/predict-url", response_model=PredictionResponse, tags=["Classification"])
def predict_url(request: URLRequest):
    """
    Scrape a news article from a URL and classify it.

    Supports: BBC, NDTV, The Hindu, Reuters, Indian Express, and most
    other major news sites. Paywalled articles will return an error.
    """
    try:
        from newspaper import Article, Config
        cfg = Config()
        cfg.fetch_images      = False
        cfg.request_timeout   = 15
        cfg.browser_user_agent = (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
        )
        article = Article(request.url, config=cfg)
        article.download()
        article.parse()

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not scrape URL: {str(e)}"
        )

    title = article.title or ''
    text  = article.text or ''

    if len(text.split()) < 50:
        raise HTTPException(
            status_code=422,
            detail="Article too short after scraping. May be paywalled."
        )

    return build_prediction_response(
        title = clean_text(title),
        text  = clean_text(text),
        url   = request.url,
    )


@app.post("/feedback", tags=["Feedback"])
def submit_feedback(request: FeedbackRequest):
    """Submit a label correction for future model retraining."""
    if not request.corrected_labels:
        raise HTTPException(
            status_code=422,
            detail="corrected_labels cannot be empty."
        )

    invalid = [l for l in request.corrected_labels if l not in CATEGORIES]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid labels: {invalid}. Must be from: {CATEGORIES}"
        )

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO corrections VALUES (?,?,?,?,?)",
        (
            None,
            request.text[:1000],
            str(request.predicted_labels),
            str(request.corrected_labels),
            datetime.now().isoformat(),
        )
    )
    conn.commit()
    count = conn.execute(
        "SELECT COUNT(*) FROM corrections"
    ).fetchone()[0]
    conn.close()

    return {
        "status":            "saved",
        "total_corrections": count,
        "message":           f"Correction saved. {count} total corrections collected."
    }


@app.get("/feedback/count", tags=["Feedback"])
def get_feedback_count():
    """Return total number of user corrections collected."""
    try:
        conn  = get_db_connection()
        count = conn.execute(
            "SELECT COUNT(*) FROM corrections"
        ).fetchone()[0]
        conn.close()
        return {"count": count}
    except Exception:
        return {"count": 0}


@app.get("/feedback/export", tags=["Feedback"])
def export_feedback():
    """Export all collected corrections (for retraining)."""
    try:
        conn = get_db_connection()
        rows = conn.execute(
            "SELECT text, predicted, corrected, ts FROM corrections"
        ).fetchall()
        conn.close()

        return {
            "count": len(rows),
            "corrections": [
                {
                    "text":      r[0],
                    "predicted": r[1],
                    "corrected": r[2],
                    "timestamp": r[3],
                }
                for r in rows
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))