"""
NewsLens 3.0 — Shared Embedding Service
Provides a unified singleton embedder (all-MiniLM-L6-v2) across the application,
preventing duplicate model instances from consuming memory and slow loading.
"""

from typing import Optional

_SHARED_EMBEDDER = None


def get_shared_embedder():
    """
    Singleton sentence-transformers embedder (all-MiniLM-L6-v2).
    Shared across micro-retrieval, local KB, and vector catalogue indexing.
    """
    global _SHARED_EMBEDDER
    if _SHARED_EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Lightweight 384-dim embedder
            _SHARED_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[Shared Embedder Warning] Could not load SentenceTransformer: {e}")
            _SHARED_EMBEDDER = None
    return _SHARED_EMBEDDER

