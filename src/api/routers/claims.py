import uuid
from fastapi import APIRouter, HTTPException
from src.api.schemas import ClaimExtractionRequest, ClaimVerificationRequest, ClaimResponse
from src.graph.nodes.ingest import segment_sentences, clean_text
from src.graph.nodes.claims import score_checkworthiness, normalize_query
from src.graph.nodes.verify import evaluate_stance, aggregate_verdict
from src.web_retrieval.factory import get_web_search_provider

router = APIRouter(prefix="", tags=["Claims & Verification"])


@router.post("/claims/extract", summary="Extract check-worthy factual claims from text")
def extract_claims(req: ClaimExtractionRequest):
    """
    Extracts and ranks check-worthy factual claims from input text.
    Filters out opinions, rhetorical questions, and returns normalized queries.
    """
    clean_b = clean_text(req.text)
    clean_t = clean_text(req.title or "")
    full_text = f"{clean_t}. {clean_b}" if clean_t else clean_b
    sentences = segment_sentences(full_text)

    scored_claims = []
    for idx, sent in enumerate(sentences):
        score = score_checkworthiness(sent)
        if score >= 0.40:
            scored_claims.append({
                "claim_id": str(uuid.uuid4()),
                "sentence_idx": idx,
                "text": sent,
                "checkworthiness_score": score,
                "normalized_query": normalize_query(sent),
            })

    scored_claims.sort(key=lambda x: x["checkworthiness_score"], reverse=True)
    return {
        "total_sentences": len(sentences),
        "checkworthy_claims_count": len(scored_claims),
        "claims": scored_claims[:5],
    }


@router.post("/verify", summary="Verify a standalone claim using live web evidence")
def verify_claim(req: ClaimVerificationRequest):
    """
    Verifies a specific factual claim against live web sources:
    - Queries search provider (DuckDuckGo or Tavily)
    - Performs NLI stance evaluation (supports, refutes, neutral)
    - Aggregates evidence under conservative policy: Supported, Refuted, Mixed, or NEI
    - Returns source diversity metrics.
    """
    if not req.claim or len(req.claim.strip()) < 10:
        raise HTTPException(status_code=422, detail="Claim text too short for verification.")

    provider = get_web_search_provider(preferred=req.search_provider or "auto")
    query = normalize_query(req.claim)

    raw_evidence = provider.search(query=query, max_results=5)
    formatted_evidence = []

    for ev in raw_evidence:
        stance, nli_conf = evaluate_stance(req.claim, ev.passage)
        formatted_evidence.append({
            "url": ev.url,
            "title": ev.title,
            "domain": ev.domain,
            "source_group": ev.source_group,
            "source_type": ev.source_type,
            "passage": ev.passage,
            "stance": stance,
            "nli_score": nli_conf,
            "retrieved_at": ev.retrieved_at,
            "provider": ev.provider,
        })

    verdict, confidence, diversity = aggregate_verdict(formatted_evidence)

    return {
        "claim": req.claim,
        "normalized_query": query,
        "verdict": verdict,
        "confidence": confidence,
        "evidence_count": len(formatted_evidence),
        "source_diversity_summary": diversity,
        "evidence": formatted_evidence,
        "limitations": [
            "Verdicts reflect retrieved evidence, not absolute truth.",
            "NEI indicates insufficient or inconclusive evidence at retrieval time, not that the claim is false."
        ],
    }

