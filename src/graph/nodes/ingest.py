import re
import uuid
from datetime import datetime
from src.graph.state import NewsLensState


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def segment_sentences(text: str) -> list[str]:
    """Segment text into clean, non-empty sentences."""
    if not text:
        return []
    # Clean sentence splitter that preserves abbreviations reasonably
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 15]
    return sentences


def ingest_node(state: NewsLensState) -> dict:
    """Ingestion & normalization node for LangGraph."""
    article_id = state.get("article_id") or str(uuid.uuid4())
    raw_title = state.get("raw_title", "")
    raw_text = state.get("raw_text", "")

    clean_t = clean_text(raw_title)
    clean_b = clean_text(raw_text)

    # Combined full text for sentence splitting
    full_text = f"{clean_t}. {clean_b}" if clean_t else clean_b
    sentences = segment_sentences(full_text)
    word_count = len(clean_b.split())

    steps = list(state.get("execution_steps", []))
    steps.append("ingest")

    return {
        "article_id": article_id,
        "clean_title": clean_t,
        "clean_text": clean_b,
        "sentences": sentences,
        "word_count": word_count,
        "execution_steps": steps,
        "processed_at": datetime.utcnow().isoformat(),
        "limitations": [
            "NewsLens 3.0 provides decision support and evidence tracing; it does not determine absolute truth.",
            "NEI indicates insufficient or inconclusive evidence at retrieval time, not that a claim is false.",
            "Source diversity is an informational signal, not a guarantee of factual accuracy."
        ],
    }

