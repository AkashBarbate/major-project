import os
import json
import re
import numpy as np
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from src.web_retrieval.provider_base import EvidenceResult


class LocalEvidenceKB:
    """
    Curated Local Evidence Knowledge Base for NewsLens 3.0:
    - Hybrid lexical (BM25) + dense (MiniLM embeddings) retrieval
    - Provides reproducible fact-check evaluation independent of changing live web results
    - Tracks provenance, publishers, and timestamps.
    """

    def __init__(self, json_path: Optional[str] = None):
        if json_path is None:
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            json_path = os.path.join(root, "data", "curated_kb", "evidence_snippets.json")

        self.json_path = json_path
        self.snippets: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.dense_embeddings: Optional[np.ndarray] = None
        self._embedder = None
        self._load_corpus()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower())

    def _load_corpus(self):
        if not os.path.exists(self.json_path):
            print(f"[Local KB Warning] File not found: {self.json_path}")
            return

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.snippets = data.get("snippets", [])

        if not self.snippets:
            return

        # Initialize BM25 corpus
        tokenized_corpus = [
            self._tokenize(f"{s.get('claim_topic', '')} {s.get('passage', '')}")
            for s in self.snippets
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Initialize Dense Embeddings using shared embedder and disk cache
        try:
            from src.taxonomy.embedding_service import get_shared_embedder
            self._embedder = get_shared_embedder()
            cache_file = os.path.join(os.path.dirname(self.json_path), "kb_embeddings.npy")

            if os.path.exists(cache_file):
                cached = np.load(cache_file)
                if len(cached) == len(self.snippets):
                    self.dense_embeddings = cached
                else:
                    passages = [f"{s.get('claim_topic', '')}: {s.get('passage', '')}" for s in self.snippets]
                    self.dense_embeddings = self._embedder.encode(passages, normalize_embeddings=True)
                    np.save(cache_file, self.dense_embeddings)
            elif self._embedder is not None:
                passages = [f"{s.get('claim_topic', '')}: {s.get('passage', '')}" for s in self.snippets]
                self.dense_embeddings = self._embedder.encode(passages, normalize_embeddings=True)
                np.save(cache_file, self.dense_embeddings)
            else:
                self.dense_embeddings = None
        except Exception as e:
            print(f"[Local KB Warning] Could not load embedder for dense KB: {e}")
            self.dense_embeddings = None

    def search(self, query: str, top_k: int = 3, threshold: float = 0.35) -> List[EvidenceResult]:
        """
        Perform hybrid BM25 + dense retrieval on curated evidence knowledge base.
        """
        results: List[EvidenceResult] = []
        if not self.snippets or not query.strip():
            return results

        # 1. Lexical BM25 Scores
        q_tokens = self._tokenize(query)
        bm25_scores = np.zeros(len(self.snippets))
        if self.bm25 and q_tokens:
            raw_bm25 = self.bm25.get_scores(q_tokens)
            max_bm = np.max(raw_bm25) if len(raw_bm25) > 0 and np.max(raw_bm25) > 0 else 1.0
            bm25_scores = raw_bm25 / max_bm

        # 2. Dense Semantic Scores
        dense_scores = np.zeros(len(self.snippets))
        if self._embedder is not None and self.dense_embeddings is not None:
            q_emb = self._embedder.encode([query], normalize_embeddings=True)[0]
            dense_scores = np.dot(self.dense_embeddings, q_emb)

        # 3. Hybrid Combination (50% lexical BM25 + 50% dense cosine)
        hybrid_scores = 0.45 * bm25_scores + 0.55 * dense_scores

        ranked_indices = np.argsort(hybrid_scores)[::-1]

        for idx in ranked_indices[:top_k]:
            h_score = float(hybrid_scores[idx])
            if h_score >= threshold:
                item = self.snippets[idx]
                results.append(
                    EvidenceResult(
                        url=item.get("source_url", ""),
                        title=f"[Local KB] {item.get('claim_topic', 'Curated Fact-Check')}",
                        passage=item.get("passage", ""),
                        domain=item.get("domain", "local.kb"),
                        source_group=item.get("source_group", "Curated Local KB"),
                        source_type=item.get("source_type", "fact_checker"),
                        score=round(h_score, 4),
                        published_at=item.get("published_at"),
                        provider="local_kb",
                    )
                )

        return results


# Singleton instance
_LOCAL_KB_INSTANCE = None

def get_local_evidence_kb() -> LocalEvidenceKB:
    global _LOCAL_KB_INSTANCE
    if _LOCAL_KB_INSTANCE is None:
        _LOCAL_KB_INSTANCE = LocalEvidenceKB()
    return _LOCAL_KB_INSTANCE

