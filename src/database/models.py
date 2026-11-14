import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Text,
    Float,
    Integer,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import relationship
from .session import Base


class Article(Base):
    __tablename__ = "articles"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(Text, nullable=False)
    body = Column(Text, nullable=False)
    url = Column(String(1024), nullable=True)
    publisher = Column(String(255), nullable=True)
    published_at = Column(DateTime, nullable=True)
    ingested_at = Column(DateTime, default=datetime.utcnow)

    # Credibility outputs
    credibility_score = Column(Float, nullable=True)
    credibility_label = Column(String(50), nullable=True)
    credibility_signals = Column(JSON, nullable=True)

    # Classification outputs
    macro_predictions = Column(JSON, nullable=True)  # list of {label, score, threshold, predicted}
    micro_labels = Column(JSON, nullable=True)       # list of {label, score}
    ood_status = Column(JSON, nullable=True)         # {ood_score, review_required, proposed_cluster}

    # Token attribution cache
    token_attributions = Column(JSON, nullable=True)

    status = Column(String(50), default="processed")

    # Relationships
    claims = relationship("Claim", back_populates="article", cascade="all, delete-orphan")
    ood_reviews = relationship("OODReviewItem", back_populates="article", cascade="all, delete-orphan")


class Claim(Base):
    __tablename__ = "claims"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    article_id = Column(String(36), ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    sentence_index = Column(Integer, nullable=False)
    claim_text = Column(Text, nullable=False)
    checkworthiness_score = Column(Float, default=0.0)
    normalized_query = Column(Text, nullable=True)

    # Verification outcome
    verdict = Column(String(50), default="NEI")  # Supported, Refuted, Mixed, NEI
    confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    article = relationship("Article", back_populates="claims")
    evidence = relationship("EvidenceSnippet", back_populates="claim", cascade="all, delete-orphan")


class EvidenceSnippet(Base):
    __tablename__ = "evidence_snippets"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    claim_id = Column(String(36), ForeignKey("claims.id", ondelete="CASCADE"), nullable=False)
    source_url = Column(String(1024), nullable=False)
    publisher = Column(String(255), nullable=True)
    domain = Column(String(255), nullable=True)
    source_group = Column(String(255), nullable=True)
    source_type = Column(String(100), default="reporting")  # official, fact_checker, reporting, academic
    title = Column(Text, nullable=True)
    passage = Column(Text, nullable=False)

    # NLI & relevance
    stance = Column(String(50), default="neutral")  # supports, refutes, neutral
    nli_score = Column(Float, default=0.0)
    relevance_score = Column(Float, default=0.0)
    retrieved_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    claim = relationship("Claim", back_populates="evidence")


class MicroCatalogueEntry(Base):
    __tablename__ = "micro_catalogue"

    id = Column(String(50), primary_key=True)
    name = Column(String(100), unique=True, index=True, nullable=False)
    macro_category = Column(String(100), index=True, nullable=False)
    definition = Column(Text, nullable=False)
    aliases = Column(JSON, default=list)
    seed_examples = Column(JSON, default=list)
    status = Column(String(20), default="active")  # active, deprecated, pending_review
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)


class OODReviewItem(Base):
    __tablename__ = "ood_review_queue"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    article_id = Column(String(36), ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    headline = Column(Text, nullable=False)
    ood_score = Column(Float, nullable=False)
    nearest_micro_label = Column(String(100), nullable=True)
    similarity = Column(Float, default=0.0)
    proposed_name = Column(String(100), nullable=True)
    cluster_id = Column(String(50), nullable=True)
    status = Column(String(50), default="pending")  # pending, approved, rejected
    reviewed_by = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    article = relationship("Article", back_populates="ood_reviews")


class HumanAnnotation(Base):
    __tablename__ = "human_annotations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    entity_type = Column(String(50), nullable=False)  # article, claim, evidence_pair
    entity_id = Column(String(36), nullable=False)
    annotator = Column(String(100), nullable=False)
    label = Column(String(100), nullable=False)
    confidence = Column(Float, default=1.0)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

