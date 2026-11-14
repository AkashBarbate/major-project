import re
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from src.database.session import get_db
from src.database.models import Article, Claim, EvidenceSnippet
from src.api.schemas import (
    ArticleAnalysisRequest,
    URLAnalysisRequest,
    ArticleAnalysisResponse,
)
from src.graph.pipeline import newslens_pipeline
from src.web_retrieval.scraper import extract_article_content

router = APIRouter(prefix="/articles", tags=["Articles & Analysis"])


def run_pipeline_for_article(title: str, text: str, url: str = None, publisher: str = None) -> dict:
    """Executes the compiled LangGraph pipeline."""
    initial_state = {
        "raw_title": title or "",
        "raw_text": text or "",
        "url": url,
        "publisher": publisher,
    }
    final_state = newslens_pipeline.invoke(initial_state)

    predicted_macros = [
        m["label"] for m in final_state.get("macro_predictions", []) if m["predicted"]
    ]

    return {
        "article_id": final_state.get("article_id"),
        "title": final_state.get("clean_title") or title,
        "url": url,
        "publisher": publisher,
        "macro_labels": final_state.get("macro_predictions", []),
        "predicted_macro_labels": predicted_macros,
        "macro_confidence": final_state.get("macro_confidence", 0.5),
        "is_macro_uncertain": final_state.get("is_macro_uncertain", False),
        "micro_labels": final_state.get("micro_labels", []),
        "topic_status": {
            "ood_score": final_state.get("ood_score", 0.0),
            "review_required": final_state.get("ood_review_required", False),
            "proposed_cluster": final_state.get("proposed_cluster"),
        },
        "credibility_score": final_state.get("credibility_score"),
        "credibility_label": final_state.get("credibility_label"),
        "credibility_signals": final_state.get("credibility_signals", []),
        "claims": final_state.get("claims", []),
        "source_diversity_summary": final_state.get("source_diversity_summary", {}),
        "limitations": final_state.get("limitations", []),
        "processed_at": final_state.get("processed_at", ""),
        "execution_steps": final_state.get("execution_steps", []),
    }


@router.post("", response_model=ArticleAnalysisResponse, summary="Analyze news article text via LangGraph")
def analyze_article(req: ArticleAnalysisRequest):
    """
    Primary NewsLens 3.0 analysis endpoint:
    - Ingests and normalizes article text
    - Runs calibrated RoBERTa macro classifier (8 categories)
    - Retrieves semantic micro-labels from approved catalogue
    - Evaluates OOD distance and checks novelty threshold
    - Extracts 8 transparent linguistic manipulation cues
    - Extracts top 1-3 check-worthy factual claims
    - Verifies claims against live web evidence & assigns conservative verdicts
    - Persists results in SQLite
    """
    if not req.text or len(req.text.strip()) < 25:
        raise HTTPException(status_code=422, detail="Article text too short. Minimum 25 characters required.")

    res = run_pipeline_for_article(
        title=req.title,
        text=req.text,
        url=req.url,
        publisher=req.publisher,
    )
    return res


@router.post("/url", response_model=ArticleAnalysisResponse, summary="Scrape news URL and analyze")
def analyze_article_url(req: URLAnalysisRequest):
    """Scrapes news article from a given URL and runs full NewsLens 3.0 analysis."""
    title, text = extract_article_content(req.url)

    if not text or len(text.strip()) < 25:
        raise HTTPException(
            status_code=422,
            detail="Extracted article text too short or could not be scraped. Site may be paywalled, behind Cloudflare, or blocked."
        )

    res = run_pipeline_for_article(
        title=title,
        text=text,
        url=req.url,
        publisher=None,
    )
    return res


@router.get("", summary="List recently analyzed articles")
def list_articles(limit: int = 20, db: Session = Depends(get_db)):
    """Retrieve recent analyzed articles from SQLite database."""
    articles = db.query(Article).order_by(Article.ingested_at.desc()).limit(limit).all()
    return [
        {
            "id": a.id,
            "title": a.title,
            "url": a.url,
            "publisher": a.publisher,
            "credibility_score": a.credibility_score,
            "credibility_label": a.credibility_label,
            "ingested_at": a.ingested_at.isoformat() if a.ingested_at else None,
            "claims_count": len(a.claims),
        }
        for a in articles
    ]


@router.get("/{article_id}", summary="Get full article analysis by ID")
def get_article(article_id: str, db: Session = Depends(get_db)):
    """Retrieve complete persisted analysis for a specific article ID."""
    article = db.query(Article).filter_by(id=article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    return {
        "article_id": article.id,
        "title": article.title,
        "body": article.body,
        "url": article.url,
        "publisher": article.publisher,
        "credibility_score": article.credibility_score,
        "credibility_label": article.credibility_label,
        "credibility_signals": article.credibility_signals,
        "macro_predictions": article.macro_predictions,
        "micro_labels": article.micro_labels,
        "ood_status": article.ood_status,
        "claims": [
            {
                "claim_id": c.id,
                "sentence_idx": c.sentence_index,
                "text": c.claim_text,
                "checkworthiness_score": c.checkworthiness_score,
                "verdict": c.verdict,
                "confidence": c.confidence,
                "evidence": [
                    {
                        "url": e.source_url,
                        "title": e.title,
                        "domain": e.domain,
                        "source_group": e.source_group,
                        "source_type": e.source_type,
                        "passage": e.passage,
                        "stance": e.stance,
                        "nli_score": e.nli_score,
                    }
                    for e in c.evidence
                ],
            }
            for c in article.claims
        ],
        "ingested_at": article.ingested_at.isoformat() if article.ingested_at else None,
    }

