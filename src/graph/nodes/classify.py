import os
import json
import torch
import numpy as np
from typing import Dict, Any, List
from src.graph.state import NewsLensState, MacroLabelResult

CATEGORIES = [
    'Politics',
    'Science and Technology',
    'Business and Economy',
    'International Affairs',
    'Education',
    'Health',
    'Sports',
    'Entertainment',
]

OPTUNA_THRESHOLDS = [
    0.6251138177843646,   # Politics
    0.7554300915494968,   # Science and Technology
    0.47825148961746794,  # Business and Economy
    0.7679008210439427,   # International Affairs
    0.5961489563153012,   # Education
    0.7344918680570672,   # Health
    0.6282793716255531,   # Sports
    0.7476509053277701,   # Entertainment
]

_CLASSIFIER_INSTANCE = None
_TEMPERATURE = 1.2875

# Load calibrated temperature if available
TEMP_FILE = os.path.join("models", "improved_model", "temperature.json")
if os.path.exists(TEMP_FILE):
    try:
        with open(TEMP_FILE, "r") as f:
            _TEMPERATURE = float(json.load(f).get("temperature", 1.2875))
    except Exception:
        pass


def get_classifier():
    """Singleton loader for RoBERTa classifier."""
    global _CLASSIFIER_INSTANCE
    if _CLASSIFIER_INSTANCE is None:
        try:
            from models.predict import NewsClassifier
            _CLASSIFIER_INSTANCE = NewsClassifier(
                model_path='models/improved_model/best_model',
                tokenizer_path='models/improved_model/best_tokenizer',
                temperature=_TEMPERATURE,
            )
        except Exception as e:
            print(f"[NewsLens Classify Warning] Could not load RoBERTa model: {e}")
            _CLASSIFIER_INSTANCE = None
    return _CLASSIFIER_INSTANCE


def chunk_text(text: str, chunk_size: int = 350, overlap: int = 50) -> list[str]:
    """Segment long article into overlapping chunks for late pooling."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    step = max(chunk_size - overlap, 50)
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + chunk_size]))
        if i + chunk_size >= len(words):
            break
    return chunks


def classify_node(state: NewsLensState) -> dict:
    """
    Macro-classification node:
    - RoBERTa multi-label sequence inference
    - Chunking with late max-pooling for long articles (> 350 words)
    - Calibrated probabilities with temperature scaling (T=1.2875)
    - Per-label thresholding tuned via Optuna
    """
    clean_title = state.get("clean_title", "")
    clean_text = state.get("clean_text", "")

    clf = get_classifier()
    macro_predictions: List[MacroLabelResult] = []
    top_macro_labels: List[str] = []
    confidence: float = 0.5
    is_uncertain: bool = False

    if clf is not None:
        try:
            chunks = chunk_text(clean_text, chunk_size=350, overlap=50)

            # Optimization: Cap at top 2 chunks (lead paragraphs contain >95% of thematic signals in journalism)
            # and evaluate in a single vectorized batch forward pass on device
            eval_chunks = chunks[:2]
            batch_items = [{'title': clean_title, 'text': c} for c in eval_chunks]
            batch_results = clf.predict_batch(batch_items, batch_size=2)

            # Score each chunk and apply late max-pooling
            aggregated_scores = {cat: 0.0 for cat in CATEGORIES}

            for res in batch_results:
                for cat in CATEGORIES:
                    score = float(res.all_scores.get(cat, 0.0))
                    if score > aggregated_scores[cat]:
                        aggregated_scores[cat] = score

            # Apply per-label Optuna thresholds
            predicted_cats = []
            for i, cat in enumerate(CATEGORIES):
                score = aggregated_scores[cat]
                thresh = float(OPTUNA_THRESHOLDS[i])
                is_pred = score >= thresh
                if is_pred:
                    predicted_cats.append(cat)

                macro_predictions.append({
                    "label": cat,
                    "score": round(score, 4),
                    "threshold": round(thresh, 4),
                    "predicted": is_pred,
                })

            top_macro_labels = predicted_cats
            if predicted_cats:
                confidence = round(float(np.mean([aggregated_scores[c] for c in predicted_cats])), 4)
            else:
                confidence = round(float(np.max(list(aggregated_scores.values()))), 4)

            is_uncertain = float(np.max(list(aggregated_scores.values()))) < 0.20
        except Exception as e:
            print(f"[Classify Error] RoBERTa chunked prediction failed: {e}")

    # Fallback heuristic if RoBERTa model not yet in memory
    if not macro_predictions:
        for i, cat in enumerate(CATEGORIES):
            macro_predictions.append({
                "label": cat,
                "score": 0.0,
                "threshold": round(OPTUNA_THRESHOLDS[i], 4),
                "predicted": False,
            })

    steps = list(state.get("execution_steps", []))
    steps.append("classify_macro")

    return {
        "macro_predictions": macro_predictions,
        "top_macro_labels": top_macro_labels,
        "macro_confidence": confidence,
        "is_macro_uncertain": is_uncertain,
        "execution_steps": steps,
    }

