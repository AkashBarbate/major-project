import re
import uuid
from typing import List, Dict, Any
from src.graph.state import NewsLensState, ClaimRecord

FACTUAL_VERBS = {
    "announced", "reported", "stated", "claimed", "passed", "approved", "rejected",
    "increased", "decreased", "signed", "launched", "unveiled", "arrested", "convicted",
    "ruled", "found", "discovered", "tested", "published", "confirmed", "allocated",
    "raised", "slashed", "imposed", "lifted", "killed", "injured", "hit", "acquired"
}

OPINION_MARKERS = {
    "i think", "i believe", "in my view", "in my opinion", "arguably", "perhaps",
    "maybe", "should be", "ought to", "what if", "we must hope", "seems to me"
}


def score_checkworthiness(sentence: str) -> float:
    """
    Heuristic check-worthiness scorer:
    Scores factual assertions containing entities, numerical metrics, dates, and event verbs.
    Penalizes speculative, opinionated, or interrogative sentences.
    """
    s_lower = sentence.lower().strip()

    # Penalize questions, extreme short/long sentences, and explicit opinion markers
    if sentence.endswith("?") or len(sentence.split()) < 7 or len(sentence.split()) > 60:
        return 0.1
    for op in OPINION_MARKERS:
        if op in s_lower:
            return 0.15

    score = 0.3  # Base score

    # 1. Numerical & statistical metrics (e.g. 25%, $50 million, 10,000, 4.5)
    if re.search(r"\b\d+([.,]\d+)?\s*(%|percent|billion|million|crore|lakh|basis points|dollars|rupees|km|kg)?\b", s_lower):
        score += 0.25

    # 2. Date and temporal markers (e.g. 2024, Friday, yesterday, October)
    if re.search(r"\b(20\d\d|january|february|march|april|may|june|july|august|september|october|november|december|yesterday|last week|on monday|on tuesday|on wednesday|on thursday|on friday)\b", s_lower):
        score += 0.15

    # 3. Capitalized named entities (People, Organizations, Governments)
    caps_entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", sentence)
    if len(caps_entities) >= 2:
        score += 0.20

    # 4. Factual action / attribution verbs
    words = set(re.findall(r"\b[a-z]+\b", s_lower))
    if words.intersection(FACTUAL_VERBS):
        score += 0.15

    return round(float(min(score, 0.98)), 3)


def normalize_query(sentence: str) -> str:
    """Normalize check-worthy sentence into a search-effective query."""
    # Strip introductory journalistic attribution (e.g. 'According to reports, ...')
    query = re.sub(r"^(according to [^,]+,|sources said|in a statement,|reports indicate that)\s*", "", sentence, flags=re.I)
    # Strip quotes
    query = query.replace('"', '').replace("'", "").strip()
    # Limit query length to first 18 words to prevent search engine truncation
    tokens = query.split()
    return " ".join(tokens[:18])


def claim_extraction_node(state: NewsLensState) -> dict:
    """
    Extracts top 1–3 check-worthy claims from the article text.
    Preserves original sentence text and marks article sentence index.
    """
    sentences = state.get("sentences", [])
    scored_claims = []

    for idx, sent in enumerate(sentences):
        # Ignore headlines if identical or ultra short intro
        if len(sent.split()) < 7:
            continue
        c_score = score_checkworthiness(sent)
        if c_score >= 0.45:
            scored_claims.append({
                "claim_id": str(uuid.uuid4()),
                "sentence_idx": idx,
                "text": sent,
                "checkworthiness_score": c_score,
                "normalized_query": normalize_query(sent),
                "verdict": "NEI",
                "confidence": 0.0,
                "evidence": [],
            })

    # Sort descending by check-worthiness score, keep top claims (default 2 for high performance)
    scored_claims.sort(key=lambda x: x["checkworthiness_score"], reverse=True)
    max_k = int(state.get("max_claims", 2))
    top_claims: List[ClaimRecord] = scored_claims[:max_k]

    steps = list(state.get("execution_steps", []))
    steps.append("claim_extraction")

    return {
        "claims": top_claims,
        "has_checkworthy_claims": len(top_claims) > 0,
        "execution_steps": steps,
    }

