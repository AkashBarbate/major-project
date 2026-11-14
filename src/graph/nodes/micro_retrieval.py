import os
import json
import uuid
import numpy as np
from typing import List, Dict, Any, Optional
from src.graph.state import NewsLensState, MicroLabelResult
from src.database.session import SessionLocal
from src.database.models import MicroCatalogueEntry, OODReviewItem
from src.taxonomy.llm_suggester import discover_micro_labels_with_llm

import os
import re
import json
import uuid
import numpy as np
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from src.graph.state import NewsLensState, MicroLabelResult
from src.database.session import SessionLocal
from src.database.models import MicroCatalogueEntry, OODReviewItem
from src.taxonomy.llm_suggester import discover_micro_labels_with_llm
from src.taxonomy.embedding_service import get_shared_embedder
from src.graph.nodes.verify import get_nli_scorer

_CATALOGUE_CACHE = []
_LABEL_EMBEDDINGS = None
_CATALOGUE_BM25 = None


def get_embedder():
    """Returns the shared sentence-transformers embedder instance."""
    return get_shared_embedder()


def _tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 lexical matching."""
    return re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower())


def load_catalogue():
    """Load approved micro-labels with disk-cached dense embeddings and BM25 index."""
    global _CATALOGUE_CACHE, _LABEL_EMBEDDINGS, _CATALOGUE_BM25
    if _CATALOGUE_CACHE and _LABEL_EMBEDDINGS is not None and _CATALOGUE_BM25 is not None:
        return _CATALOGUE_CACHE

    db = SessionLocal()
    try:
        entries = db.query(MicroCatalogueEntry).filter_by(status="active").all()
        if entries:
            _CATALOGUE_CACHE = [
                {
                    "id": e.id,
                    "name": e.name,
                    "macro_category": e.macro_category,
                    "definition": e.definition,
                    "aliases": e.aliases or [],
                    "seed_examples": e.seed_examples or [],
                }
                for e in entries
            ]
    except Exception as e:
        print(f"[Catalogue Load DB Warning] {e}")
    finally:
        db.close()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    cat_dir = os.path.join(root, "data", "catalogue")

    if not _CATALOGUE_CACHE:
        json_path = os.path.join(cat_dir, "micro_labels.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                _CATALOGUE_CACHE = json.load(f).get("labels", [])

    # Initialize BM25 Lexical Index
    if _CATALOGUE_CACHE:
        tokenized_corpus = [
            _tokenize(f"{c['name']} {' '.join(c.get('aliases', []))} {c['definition']}")
            for c in _CATALOGUE_CACHE
        ]
        _CATALOGUE_BM25 = BM25Okapi(tokenized_corpus)

    # Load or compute dense catalogue embeddings with disk caching
    embedder = get_embedder()
    cache_path = os.path.join(cat_dir, "catalogue_embeddings.npy")

    if _CATALOGUE_CACHE and _LABEL_EMBEDDINGS is None:
        if os.path.exists(cache_path):
            try:
                cached_emb = np.load(cache_path)
                if len(cached_emb) == len(_CATALOGUE_CACHE):
                    _LABEL_EMBEDDINGS = cached_emb
            except Exception:
                _LABEL_EMBEDDINGS = None

        if _LABEL_EMBEDDINGS is None and embedder is not None:
            texts = [f"{c['name']}: {c['definition']} " + " ".join(c.get("aliases", [])) for c in _CATALOGUE_CACHE]
            _LABEL_EMBEDDINGS = embedder.encode(texts, normalize_embeddings=True)
            try:
                np.save(cache_path, _LABEL_EMBEDDINGS)
            except Exception:
                pass

    return _CATALOGUE_CACHE


def register_micro_label_in_vector_db(
    name: str,
    macro_category: str,
    definition: str,
    aliases: List[str],
    seed_example: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Persists a novel micro-label to SQLite and dynamically updates both
    the dense vector index and the BM25 lexical index.
    """
    global _CATALOGUE_CACHE, _LABEL_EMBEDDINGS, _CATALOGUE_BM25
    clean_name = name.strip().lower()
    if not clean_name:
        return None

    db = SessionLocal()
    entry_dict = None
    try:
        existing = db.query(MicroCatalogueEntry).filter_by(name=clean_name).first()
        if existing:
            if seed_example and seed_example not in (existing.seed_examples or []):
                ex = list(existing.seed_examples or [])
                ex.append(seed_example)
                existing.seed_examples = ex
                db.commit()
            entry_dict = {
                "id": existing.id,
                "name": existing.name,
                "macro_category": existing.macro_category,
                "definition": existing.definition,
                "aliases": existing.aliases or [],
                "seed_examples": existing.seed_examples or [],
            }
        else:
            new_id = f"micro_{uuid.uuid4().hex[:8]}"
            new_entry = MicroCatalogueEntry(
                id=new_id,
                name=clean_name,
                macro_category=macro_category,
                definition=definition,
                aliases=aliases or [],
                seed_examples=[seed_example] if seed_example else [],
                status="active",
            )
            db.add(new_entry)
            db.commit()
            db.refresh(new_entry)
            entry_dict = {
                "id": new_entry.id,
                "name": new_entry.name,
                "macro_category": new_entry.macro_category,
                "definition": new_entry.definition,
                "aliases": new_entry.aliases or [],
                "seed_examples": new_entry.seed_examples or [],
            }

        # Dynamically update in-memory catalogue cache, dense vector index, and BM25 index
        if entry_dict:
            cached_ids = {c["id"] for c in _CATALOGUE_CACHE}
            if entry_dict["id"] not in cached_ids:
                _CATALOGUE_CACHE.append(entry_dict)
                embedder = get_embedder()
                if embedder is not None:
                    text_to_embed = f"{entry_dict['name']}: {entry_dict['definition']} " + " ".join(entry_dict.get("aliases", []))
                    new_emb = embedder.encode([text_to_embed], normalize_embeddings=True)
                    if _LABEL_EMBEDDINGS is not None:
                        _LABEL_EMBEDDINGS = np.vstack([_LABEL_EMBEDDINGS, new_emb])
                    else:
                        _LABEL_EMBEDDINGS = new_emb

                # Rebuild BM25 with newly added label
                tokenized_corpus = [
                    _tokenize(f"{c['name']} {' '.join(c.get('aliases', []))} {c['definition']}")
                    for c in _CATALOGUE_CACHE
                ]
                _CATALOGUE_BM25 = BM25Okapi(tokenized_corpus)

        return entry_dict
    except Exception as e:
        db.rollback()
        print(f"[Register Vector DB Error] {e}")
        return None
    finally:
        db.close()


def micro_retrieval_node(state: NewsLensState) -> dict:
    """
    Hybrid (BM25 + Semantic Search) Micro-Label Retrieval with Cross-Encoder Reranker:
    1. First stage: BM25 lexical scoring + MiniLM dense cosine similarity.
    2. Linear hybrid fusion selects Top 10 candidate labels.
    3. Second stage: CrossEncoder reranks the 10 candidates with joint token attention.
    4. Flexible multi-label selection retains all confident topics (up to a realistic limit of 6).
    5. If matched labels <= 1, routes context to Gemini LLM for novel topic discovery.
    """
    global _CATALOGUE_BM25
    title = state.get("clean_title", "")
    text = state.get("clean_text", "")
    top_macros = set(state.get("top_macro_labels", []))

    embedder = get_embedder()
    catalogue = load_catalogue()

    retrieved_micro: List[MicroLabelResult] = []
    ood_score = 0.0
    review_required = False
    proposed_cluster = None

    query_text = f"{title}. {' '.join(text.split()[:160])}"

    if embedder is not None and catalogue and _LABEL_EMBEDDINGS is not None:
        # --- Stage 1A: Dense Semantic Similarities ---
        doc_emb = embedder.encode([query_text], normalize_embeddings=True)[0]
        dense_sims = np.dot(_LABEL_EMBEDDINGS, doc_emb)

        # --- Stage 1B: Lexical BM25 Scoring ---
        q_tokens = _tokenize(query_text)
        if _CATALOGUE_BM25 is not None and q_tokens:
            bm25_raw = _CATALOGUE_BM25.get_scores(q_tokens)
            max_bm = float(np.max(bm25_raw)) if len(bm25_raw) > 0 and np.max(bm25_raw) > 0 else 1.0
            bm25_norm = bm25_raw / max_bm
        else:
            bm25_norm = np.zeros(len(catalogue))

        # --- Stage 1C: Linear Hybrid Fusion (35% lexical BM25 + 65% dense semantic) ---
        hybrid_scores = 0.35 * bm25_norm + 0.65 * dense_sims
        max_hybrid = float(np.max(hybrid_scores))
        best_idx = int(np.argmax(hybrid_scores))

        # OOD score is inverse similarity distance
        ood_score = round(float(np.clip(1.0 - max_hybrid, 0.0, 1.0)), 4)

        # --- Stage 2: Select Top 10 Candidates for Cross-Encoder Reranking ---
        top_10_indices = np.argsort(hybrid_scores)[::-1][:10]

        scorer = get_nli_scorer()
        if scorer is not None and len(top_10_indices) > 0:
            ce_pairs = [
                (
                    query_text,
                    f"{catalogue[idx]['name']} ({catalogue[idx]['macro_category']}): {catalogue[idx]['definition']} {' '.join(catalogue[idx].get('aliases', []))}"
                )
                for idx in top_10_indices
            ]
            try:
                raw_ce = scorer.predict(ce_pairs, batch_size=10)
                ce_conf = 1.0 / (1.0 + np.exp(-raw_ce))
            except Exception:
                ce_conf = np.array([hybrid_scores[idx] for idx in top_10_indices])
        else:
            ce_conf = np.array([hybrid_scores[idx] for idx in top_10_indices])

        # --- Stage 3: Unified Reranked Scoring ---
        candidate_results = []
        for rank_pos, idx in enumerate(top_10_indices):
            if idx >= len(catalogue):
                continue
            h_score = float(hybrid_scores[idx])
            c_score = float(ce_conf[rank_pos])
            # 55% Cross-Encoder joint attention + 45% Hybrid retrieval score
            final_score = round(0.55 * c_score + 0.45 * h_score, 4)
            candidate_results.append((idx, final_score, h_score))

        # Sort candidates descending by reranked score
        candidate_results.sort(key=lambda x: x[1], reverse=True)

        # --- Stage 4: Multi-Label Selection (Realistic Ceiling up to 6 labels) ---
        seen_names = set()
        for idx, final_score, h_score in candidate_results:
            cat_entry = catalogue[idx]
            parent_macro = cat_entry["macro_category"]
            c_name = cat_entry["name"].lower()

            if c_name in seen_names:
                continue

            # Multi-label inclusion criterion:
            # - Score >= 0.42 if compatible with predicted macro categories
            # - Score >= 0.52 for cross-domain high-relevance topics
            if final_score >= 0.42:
                if not top_macros or parent_macro in top_macros or final_score >= 0.52:
                    retrieved_micro.append({
                        "id": cat_entry["id"],
                        "name": cat_entry["name"],
                        "macro_category": parent_macro,
                        "score": final_score,
                        "is_llm_discovered": False,
                    })
                    seen_names.add(c_name)

        # Flag for human review if max similarity is weak or macro categories are absent
        if max_hybrid < 0.38 or (not top_macros and max_hybrid < 0.45):
            review_required = True
            proposed_cluster = {
                "article_id": state.get("article_id"),
                "headline": title[:150],
                "ood_score": ood_score,
                "nearest_candidate": catalogue[best_idx]["name"] if best_idx < len(catalogue) else "General",
                "candidate_similarity": round(max_hybrid, 4),
                "status": "pending_human_review",
            }
    else:
        # Fallback simple keyword matching if embedder isn't available
        combined_lower = f"{title} {text}".lower()
        for cat_entry in catalogue:
            for alias in cat_entry.get("aliases", []):
                if alias.lower() in combined_lower:
                    retrieved_micro.append({
                        "id": cat_entry["id"],
                        "name": cat_entry["name"],
                        "macro_category": cat_entry["macro_category"],
                        "score": 0.75,
                        "is_llm_discovered": False,
                    })
                    break
            if len(retrieved_micro) >= 6:
                break

    # --------------------------------------------------------------------------
    # LLM-Assisted Micro-Label Discovery:
    # If the article has fewer than 2 confident labels (e.g. 0 or 1), route
    # context to Gemini LLM to discover novel fine-grained labels and index them.
    # --------------------------------------------------------------------------
    if len(retrieved_micro) <= 1:
        needed_count = max(1, 3 - len(retrieved_micro))
        existing_names = [m["name"].lower() for m in retrieved_micro]

        discovered = discover_micro_labels_with_llm(
            headline=title,
            body=text[:1000],
            top_macro_categories=list(top_macros),
            existing_micro_names=existing_names,
            max_labels=needed_count,
        )

        for item in discovered:
            cname = item.get("canonical_name", "").strip().lower()
            if not cname or cname in existing_names:
                continue

            macro_cat = item.get("macro_category") or (list(top_macros)[0] if top_macros else "Science and Technology")
            definition = item.get("definition") or f"Reporting on {cname}."
            aliases = item.get("aliases") or [cname]

            # Register in SQLite catalogue and append to vector DB & BM25
            entry = register_micro_label_in_vector_db(
                name=cname,
                macro_category=macro_cat,
                definition=definition,
                aliases=aliases,
                seed_example=title,
            )

            if entry:
                retrieved_micro.append({
                    "id": entry["id"],
                    "name": entry["name"],
                    "macro_category": entry["macro_category"],
                    "score": 0.88,
                    "is_llm_discovered": True,
                })
                existing_names.append(cname)

    steps = list(state.get("execution_steps", []))
    steps.append("micro_retrieval_and_ood")

    # Realistic multi-label ceiling: return up to 6 micro-labels
    return {
        "micro_labels": retrieved_micro[:6],
        "ood_score": ood_score,
        "ood_review_required": review_required,
        "proposed_cluster": proposed_cluster,
        "execution_steps": steps,
    }
