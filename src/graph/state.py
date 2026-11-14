from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass, field


class ClaimRecord(TypedDict):
    claim_id: str
    sentence_idx: int
    text: str
    checkworthiness_score: float
    normalized_query: str
    verdict: str  # Supported, Refuted, Mixed, NEI
    confidence: float
    evidence: List[Dict[str, Any]]


class MacroLabelResult(TypedDict):
    label: str
    score: float
    threshold: float
    predicted: bool


class MicroLabelResult(TypedDict, total=False):
    id: str
    name: str
    macro_category: str
    score: float
    is_llm_discovered: Optional[bool]


class CredibilitySignalDict(TypedDict):
    name: str
    value: float
    display_value: float


class NewsLensState(TypedDict, total=False):
    # Input
    article_id: str
    raw_title: str
    raw_text: str
    url: Optional[str]
    publisher: Optional[str]

    # Preprocessed
    clean_title: str
    clean_text: str
    sentences: List[str]
    word_count: int

    # Macro Classification
    macro_predictions: List[MacroLabelResult]
    top_macro_labels: List[str]
    macro_confidence: float
    is_macro_uncertain: bool

    # Micro Classification & OOD
    micro_labels: List[MicroLabelResult]
    ood_score: float
    ood_review_required: bool
    proposed_cluster: Optional[Dict[str, Any]]

    # Credibility & Rules
    credibility_score: Optional[float]
    credibility_label: Optional[str]
    credibility_signals: List[CredibilitySignalDict]

    # Claims & Verification
    claims: List[ClaimRecord]
    has_checkworthy_claims: bool
    source_diversity_summary: Dict[str, Any]

    # Explainability
    token_attributions: Optional[List[Dict[str, Any]]]

    # Pipeline Meta
    execution_steps: List[str]
    processed_at: str
    limitations: List[str]
    error: Optional[str]

