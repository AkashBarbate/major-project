import os
import sqlite3
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from src.database.session import init_db
from src.database.seed import seed_micro_catalogue
from src.api.routers.articles import router as articles_router, run_pipeline_for_article
from src.api.routers.classification import router as classification_router
from src.api.routers.claims import router as claims_router
from src.api.routers.taxonomy import router as taxonomy_router
from src.api.routers.annotations import router as annotations_router
from src.graph.nodes.classify import CATEGORIES, OPTUNA_THRESHOLDS


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and seed catalogue on startup."""
    print("[NewsLens 3.0] Initializing SQLite database...")
    init_db()
    seed_micro_catalogue()
    print("[NewsLens 3.0] Ready.")
    yield


app = FastAPI(
    title="NewsLens 3.0 API",
    description="Evidence-Aware News Analysis & Verification System with LangGraph Orchestration",
    version="3.0.0",
    lifespan=lifespan,
)

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount NewsLens 3.0 Routers
app.include_router(articles_router)
app.include_router(classification_router)
app.include_router(claims_router)
app.include_router(taxonomy_router)
app.include_router(annotations_router)


# Backward Compatibility Schemas & Routes (Minor Project)
class LegacyArticleRequest(BaseModel):
    title: str = Field(default="", description="Article headline")
    text: str = Field(..., description="Article body text")
    url: str = Field(default="", description="Source URL (optional)")


class LegacyURLRequest(BaseModel):
    url: str = Field(..., description="Full URL of news article")


class LegacyFeedbackRequest(BaseModel):
    text: str
    predicted_labels: list[str]
    corrected_labels: list[str]


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "system": "NewsLens 3.0",
        "database": "SQLite (data/newslens.db)",
        "orchestrator": "LangGraph",
        "macro_categories": CATEGORIES,
    }


@app.post("/predict", tags=["Legacy Compatibility"])
def legacy_predict(request: LegacyArticleRequest):
    """Backward-compatible endpoint matching minor project /predict format."""
    if not request.text or len(request.text.strip()) < 20:
        raise HTTPException(status_code=422, detail="Article text too short.")

    res = run_pipeline_for_article(
        title=request.title,
        text=request.text,
        url=request.url,
    )

    return {
        "predicted_labels": res["predicted_macro_labels"],
        "label_scores": [
            {
                "label": m["label"],
                "score": m["score"],
                "threshold": m["threshold"],
                "predicted": m["predicted"],
            }
            for m in res["macro_labels"]
        ],
        "overall_confidence": res["macro_confidence"],
        "is_uncertain": res["is_macro_uncertain"],
        "credibility_score": res["credibility_score"],
        "credibility_label": res["credibility_label"],
        "credibility_signals": res["credibility_signals"],
        "title": res["title"][:200],
        "word_count": len(request.text.split()),
        "url": request.url,
        "processed_at": res["processed_at"],
        "micro_labels": res["micro_labels"],
        "claims": res["claims"],
    }


@app.post("/feedback", tags=["Feedback"])
def legacy_feedback(request: LegacyFeedbackRequest):
    """Feedback correction endpoint."""
    os.makedirs("models", exist_ok=True)
    conn = sqlite3.connect("models/feedback.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY,
            text TEXT,
            predicted TEXT,
            corrected TEXT,
            ts TEXT
        )
    """)
    conn.execute(
        "INSERT INTO corrections VALUES (?,?,?,?,?)",
        (None, request.text[:1000], str(request.predicted_labels), str(request.corrected_labels), datetime.now().isoformat())
    )
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
    conn.close()
    return {"status": "saved", "total_corrections": count}

