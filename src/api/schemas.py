from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# Request Schemas
class ArticleAnalysisRequest(BaseModel):
    title: str = Field(default="", description="Article headline")
    text: str = Field(..., description="Full article body text")
    url: Optional[str] = Field(default=None, description="Source URL (optional)")
    publisher: Optional[str] = Field(default=None, description="Publisher name (optional)")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "RBI raises repo rate by 25 basis points to control inflation",
                "text": "The Reserve Bank of India on Friday raised its benchmark lending rate by 25 basis points...",
                "url": "https://www.thehindu.com/business/economy/rbi-rate-hike",
                "publisher": "The Hindu",
            }
        }


class URLAnalysisRequest(BaseModel):
    url: str = Field(..., description="Full URL of news article to scrape and analyze")


class ClassifyOnlyRequest(BaseModel):
    title: str = Field(default="", description="Article headline")
    text: str = Field(..., description="Article body text")


class ClaimExtractionRequest(BaseModel):
    text: str = Field(..., description="Article text from which to extract check-worthy claims")
    title: Optional[str] = Field(default="", description="Article headline")


class ClaimVerificationRequest(BaseModel):
    claim: str = Field(..., description="Factual claim text to verify")
    context: Optional[str] = Field(default="", description="Surrounding context or article headline")
    search_provider: Optional[str] = Field(default="auto", description="'auto', 'duckduckgo', or 'tavily'")


class TaxonomyActionRequest(BaseModel):
    name: str = Field(..., description="Micro-label name")
    macro_category: str = Field(..., description="Associated macro category (1 of 8)")
    definition: str = Field(..., description="Clear definition of micro-label scope")
    aliases: List[str] = Field(default_factory=list, description="Common aliases and search keywords")
    seed_examples: List[str] = Field(default_factory=list, description="3-10 representative headlines")


class OODReviewActionRequest(BaseModel):
    action: str = Field(..., description="'approve' or 'reject'")
    approved_name: Optional[str] = Field(default=None, description="Approved canonical micro-label name")
    macro_category: Optional[str] = Field(default=None, description="Assigned macro category")
    reviewer: Optional[str] = Field(default="Analyst", description="Name/ID of human reviewer")


class HumanAnnotationRequest(BaseModel):
    entity_type: str = Field(..., description="'article', 'claim', or 'evidence_pair'")
    entity_id: str = Field(..., description="ID of entity being annotated")
    annotator: str = Field(..., description="Annotator name (Akash / Varun)")
    label: str = Field(..., description="Annotation decision (e.g. 'Supported', 'Technology')")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    notes: Optional[str] = Field(default="", description="Adjudication notes")


# Response Schemas
class MacroLabelScore(BaseModel):
    label: str
    score: float
    threshold: float
    predicted: bool


class MicroLabelScore(BaseModel):
    id: str
    name: str
    macro_category: str
    score: float


class CredibilitySignalResponse(BaseModel):
    name: str
    value: float
    display_value: float


class EvidenceSnippetResponse(BaseModel):
    url: str
    title: Optional[str] = None
    domain: Optional[str] = None
    source_group: Optional[str] = None
    source_type: Optional[str] = "reporting"
    passage: str
    stance: str  # supports, refutes, neutral
    nli_score: float
    retrieved_at: str


class ClaimResponse(BaseModel):
    claim_id: str
    sentence_idx: int
    text: str
    checkworthiness_score: float
    verdict: str  # Supported, Refuted, Mixed, NEI
    confidence: float
    evidence: List[EvidenceSnippetResponse]


class TopicStatusResponse(BaseModel):
    ood_score: float
    review_required: bool
    proposed_cluster: Optional[Dict[str, Any]] = None


class ArticleAnalysisResponse(BaseModel):
    article_id: str
    title: str
    url: Optional[str] = None
    publisher: Optional[str] = None
    macro_labels: List[MacroLabelScore]
    predicted_macro_labels: List[str]
    macro_confidence: float
    is_macro_uncertain: bool
    micro_labels: List[MicroLabelScore]
    topic_status: TopicStatusResponse
    credibility_score: Optional[float] = None
    credibility_label: Optional[str] = None
    credibility_signals: List[CredibilitySignalResponse]
    claims: List[ClaimResponse]
    source_diversity_summary: Dict[str, Any]
    limitations: List[str]
    processed_at: str
    execution_steps: List[str]

