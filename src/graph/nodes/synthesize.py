from typing import Dict, Any
from src.graph.state import NewsLensState
from src.database.session import SessionLocal
from src.database.models import Article, Claim, EvidenceSnippet, OODReviewItem


def synthesize_and_persist_node(state: NewsLensState) -> dict:
    """
    Final graph node:
    - Persists article analysis, claims, and evidence to SQLite
    - Finalizes JSON payload strictly matching NewsLens 3.0 Core API schema.
    """
    article_id = state.get("article_id")
    title = state.get("clean_title") or state.get("raw_title", "")
    body = state.get("clean_text") or state.get("raw_text", "")
    url = state.get("url")
    publisher = state.get("publisher")

    # Persist to SQLite
    try:
        db = SessionLocal()
        existing = db.query(Article).filter_by(id=article_id).first()
        if not existing:
            art = Article(
                id=article_id,
                title=title[:500],
                body=body,
                url=url,
                publisher=publisher,
                credibility_score=state.get("credibility_score"),
                credibility_label=state.get("credibility_label"),
                credibility_signals=state.get("credibility_signals"),
                macro_predictions=state.get("macro_predictions"),
                micro_labels=state.get("micro_labels"),
                ood_status={
                    "ood_score": state.get("ood_score", 0.0),
                    "review_required": state.get("ood_review_required", False),
                    "proposed_cluster": state.get("proposed_cluster"),
                },
                token_attributions=state.get("token_attributions"),
                status="completed",
            )
            db.add(art)
            db.flush()

            # Persist proposed OOD cluster card if flagged
            prop = state.get("proposed_cluster")
            if prop:
                ood_item = OODReviewItem(
                    article_id=article_id,
                    headline=title[:250],
                    ood_score=state.get("ood_score", 0.0),
                    nearest_micro_label=prop.get("nearest_candidate"),
                    similarity=prop.get("candidate_similarity", 0.0),
                    status="pending",
                )
                db.add(ood_item)

            for clm in state.get("claims", []):
                claim_db = Claim(
                    id=clm["claim_id"],
                    article_id=article_id,
                    sentence_index=clm.get("sentence_idx", 0),
                    claim_text=clm["text"],
                    checkworthiness_score=clm.get("checkworthiness_score", 0.0),
                    normalized_query=clm.get("normalized_query", ""),
                    verdict=clm.get("verdict", "NEI"),
                    confidence=clm.get("confidence", 0.5),
                )
                db.add(claim_db)
                db.flush()

                for ev in clm.get("evidence", []):
                    ev_db = EvidenceSnippet(
                        claim_id=clm["claim_id"],
                        source_url=ev["url"],
                        publisher=ev.get("publisher"),
                        domain=ev.get("domain"),
                        source_group=ev.get("source_group"),
                        source_type=ev.get("source_type", "reporting"),
                        title=ev.get("title"),
                        passage=ev.get("passage", ""),
                        stance=ev.get("stance", "neutral"),
                        nli_score=ev.get("nli_score", 0.5),
                    )
                    db.add(ev_db)

            db.commit()
    except Exception as e:
        print(f"[Synthesize DB Warning] Failed to persist run: {e}")
    finally:
        db.close()

    steps = list(state.get("execution_steps", []))
    steps.append("synthesize_and_persist")

    return {
        "execution_steps": steps,
    }
