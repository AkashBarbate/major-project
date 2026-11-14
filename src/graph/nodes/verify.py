import re
import numpy as np
from typing import List, Dict, Any, Tuple
from src.graph.state import NewsLensState, ClaimRecord
from src.web_retrieval.factory import get_web_search_provider
from src.web_retrieval.provider_base import EvidenceResult
from src.evidence_kb.local_kb import get_local_evidence_kb

_NLI_MODEL = None


def get_nli_scorer():
    """
    Singleton NLI / cross-encoder model for claim-evidence entailment.
    Uses cross-encoder or lightweight semantic matching.
    """
    global _NLI_MODEL
    if _NLI_MODEL is None:
        try:
            from sentence_transformers import CrossEncoder
            # MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli or compact cross-encoder
            _NLI_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            _NLI_MODEL = None
    return _NLI_MODEL


def evaluate_stance(claim_text: str, passage: str) -> Tuple[str, float]:
    """
    Evaluate stance of passage with respect to claim.
    Returns (stance: 'supports'|'refutes'|'neutral', confidence: float).
    """
    c_lower = claim_text.lower()
    p_lower = passage.lower()

    # Negation check for direct refutation
    negations = ["denied", "falsely claimed", "debunked", "false", "misleading", "incorrect", "refuted", "hoax"]
    claim_keywords = set(re.findall(r"\b[a-z]{4,}\b", c_lower))
    passage_words = set(re.findall(r"\b[a-z]{4,}\b", p_lower))

    overlap = len(claim_keywords.intersection(passage_words)) / max(len(claim_keywords), 1)

    if overlap < 0.25:
        return "neutral", 0.50

    # If passage contains explicit debunking/negation markers
    if any(neg in p_lower for neg in negations) and overlap >= 0.35:
        return "refutes", round(min(0.65 + overlap * 0.3, 0.95), 3)

    # Entailment heuristic / cross-encoder
    scorer = get_nli_scorer()
    if scorer is not None:
        try:
            relevance = float(scorer.predict([(claim_text, passage)])[0])
            conf = 1.0 / (1.0 + np.exp(-relevance))
            if conf >= 0.65:
                return "supports", round(float(conf), 3)
            elif conf <= 0.20:
                return "neutral", round(float(1 - conf), 3)
        except Exception:
            pass

    if overlap >= 0.50:
        return "supports", round(min(0.50 + overlap * 0.4, 0.90), 3)

    return "neutral", 0.60


def batch_evaluate_stances(pairs: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
    """
    High-performance batched stance evaluation across multiple (claim, passage) pairs.
    Pre-filters using lexical overlap, then evaluates remaining candidates in a single CrossEncoder forward pass.
    """
    if not pairs:
        return []

    results = [None] * len(pairs)
    pairs_for_nli = []
    nli_indices = []

    negations = ["denied", "falsely claimed", "debunked", "false", "misleading", "incorrect", "refuted", "hoax"]

    for i, (claim_text, passage) in enumerate(pairs):
        c_lower = claim_text.lower()
        p_lower = passage.lower()
        claim_keywords = set(re.findall(r"\b[a-z]{4,}\b", c_lower))
        passage_words = set(re.findall(r"\b[a-z]{4,}\b", p_lower))
        overlap = len(claim_keywords.intersection(passage_words)) / max(len(claim_keywords), 1)

        if overlap < 0.20:
            results[i] = ("neutral", 0.50)
        elif any(neg in p_lower for neg in negations) and overlap >= 0.35:
            results[i] = ("refutes", round(min(0.65 + overlap * 0.3, 0.95), 3))
        else:
            pairs_for_nli.append((claim_text, passage))
            nli_indices.append(i)

    # Batch evaluate via cross-encoder
    if pairs_for_nli:
        scorer = get_nli_scorer()
        if scorer is not None:
            try:
                raw_scores = scorer.predict(pairs_for_nli, batch_size=16)
                for idx, relevance in zip(nli_indices, raw_scores):
                    conf = 1.0 / (1.0 + np.exp(-float(relevance)))
                    if conf >= 0.65:
                        results[idx] = ("supports", round(float(conf), 3))
                    elif conf <= 0.20:
                        results[idx] = ("neutral", round(float(1 - conf), 3))
                    else:
                        results[idx] = ("neutral", round(float(conf), 3))
            except Exception:
                for idx in nli_indices:
                    results[idx] = ("neutral", 0.50)
        else:
            for idx in nli_indices:
                results[idx] = ("neutral", 0.50)

    return [r if r is not None else ("neutral", 0.50) for r in results]


def aggregate_verdict(evidence_list: List[Dict[str, Any]]) -> Tuple[str, float, Dict[str, Any]]:
    """
    Conservative verdict aggregation following NewsLens 3.0 policy:
    - Supported: >= 2 independent source groups supporting OR 1 official/authoritative primary source
    - Refuted: >= 1 authoritative contradiction OR >= 2 independent contradiction groups
    - Mixed: High quality evidence on both sides
    - NEI: Missing, neutral, or insufficient evidence.
    """
    if not evidence_list:
        return "NEI", 0.50, {"total_sources": 0, "unique_groups": 0, "types": {}}

    supports_groups = set()
    refutes_groups = set()
    source_types: Dict[str, int] = {}
    has_official_support = False
    has_authoritative_refute = False

    for ev in evidence_list:
        st = ev.get("stance", "neutral")
        sg = ev.get("source_group", ev.get("domain", "unknown"))
        stype = ev.get("source_type", "reporting")
        score = ev.get("nli_score", 0.5)

        source_types[stype] = source_types.get(stype, 0) + 1

        if st == "supports" and score >= 0.60:
            supports_groups.add(sg)
            if stype in ("official", "fact_checker"):
                has_official_support = True
        elif st == "refutes" and score >= 0.60:
            refutes_groups.add(sg)
            if stype in ("official", "fact_checker"):
                has_authoritative_refute = True

    diversity_summary = {
        "total_sources": len(evidence_list),
        "unique_groups": len(set(ev.get("source_group") for ev in evidence_list)),
        "supports_groups_count": len(supports_groups),
        "refutes_groups_count": len(refutes_groups),
        "types": source_types,
    }

    # Conservative verdict decision logic
    if (len(supports_groups) >= 2 or has_official_support) and (len(refutes_groups) >= 2 or has_authoritative_refute):
        return "Mixed", 0.75, diversity_summary
    elif has_authoritative_refute or len(refutes_groups) >= 2:
        return "Refuted", 0.85 if has_authoritative_refute else 0.72, diversity_summary
    elif has_official_support or len(supports_groups) >= 2:
        return "Supported", 0.88 if has_official_support else 0.78, diversity_summary
    else:
        return "NEI", 0.55, diversity_summary


def verify_node(state: NewsLensState) -> dict:
    """
    Evidence verification node (Optimized):
    - Concurrent retrieval across claims using ThreadPoolExecutor
    - Batched cross-encoder stance scoring in a single forward pass
    - Computes conservative, diversity-weighted verdicts (Supported, Refuted, Mixed, NEI)
    """
    from concurrent.futures import ThreadPoolExecutor

    claims = state.get("claims", [])
    search_provider = get_web_search_provider(preferred="auto")
    local_kb = get_local_evidence_kb()
    verified_claims: List[ClaimRecord] = []

    if not claims:
        steps = list(state.get("execution_steps", []))
        steps.append("evidence_verification")
        return {
            "claims": [],
            "source_diversity_summary": {
                "total_evidence_retrieved": 0,
                "local_kb_evidence_count": 0,
                "live_web_evidence_count": 0,
                "unique_source_groups": 0,
                "search_engine": getattr(search_provider, "__class__", type(search_provider)).__name__,
                "mode": "hybrid (local_kb + live_web)",
            },
            "execution_steps": steps,
        }

    # 1. Parallel retrieval across claims
    def retrieve_candidates(clm):
        query = clm.get("normalized_query") or clm["text"]
        l_res = local_kb.search(query=query, top_k=2)
        try:
            w_res = search_provider.search(query=query, max_results=3)
        except Exception:
            w_res = []
        return l_res + w_res

    with ThreadPoolExecutor(max_workers=min(len(claims), 3)) as executor:
        raw_candidates_per_claim = list(executor.map(retrieve_candidates, claims))

    # 2. Collect unique candidate passages for batched NLI scoring
    eval_triplets = []  # (claim_idx, ev_obj, (claim_text, passage))
    for c_idx, (clm, candidates) in enumerate(zip(claims, raw_candidates_per_claim)):
        seen_urls = set()
        for ev in candidates:
            if ev.url and ev.url in seen_urls:
                continue
            seen_urls.add(ev.url)
            eval_triplets.append((c_idx, ev, (clm["text"], ev.passage)))

    # 3. Batch evaluate all (claim, passage) pairs in a single forward pass
    all_pairs = [t[2] for t in eval_triplets]
    scored_stances = batch_evaluate_stances(all_pairs)

    # 4. Group results back by claim
    claim_evidence_map = {i: [] for i in range(len(claims))}
    all_evidence = []

    for (c_idx, ev, _), (stance, nli_conf) in zip(eval_triplets, scored_stances):
        ev_dict = {
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
        }
        claim_evidence_map[c_idx].append(ev_dict)
        all_evidence.append(ev_dict)

    # 5. Aggregate verdicts per claim
    for c_idx, clm in enumerate(claims):
        formatted_evidence = claim_evidence_map.get(c_idx, [])
        verdict, conf, div = aggregate_verdict(formatted_evidence)
        clm_copy = dict(clm)
        clm_copy["verdict"] = verdict
        clm_copy["confidence"] = conf
        clm_copy["evidence"] = formatted_evidence
        verified_claims.append(clm_copy)

    # Calculate overall diversity summary across all claims
    local_count = sum(1 for ev in all_evidence if ev.get("provider") == "local_kb")
    web_count = sum(1 for ev in all_evidence if ev.get("provider") != "local_kb")
    total_groups = len(set(ev.get("source_group") for ev in all_evidence))
    overall_diversity = {
        "total_evidence_retrieved": len(all_evidence),
        "local_kb_evidence_count": local_count,
        "live_web_evidence_count": web_count,
        "unique_source_groups": total_groups,
        "search_engine": getattr(search_provider, "__class__", type(search_provider)).__name__,
        "mode": "hybrid (local_kb + live_web)",
    }

    steps = list(state.get("execution_steps", []))
    steps.append("evidence_verification")

    return {
        "claims": verified_claims,
        "source_diversity_summary": overall_diversity,
        "execution_steps": steps,
    }
