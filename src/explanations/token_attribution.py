import os
from typing import List, Dict, Any, Optional

_EXPLAINER = None


def get_explainer():
    """Singleton explainer using transformers_interpret."""
    global _EXPLAINER
    if _EXPLAINER is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from transformers_interpret import SequenceClassificationExplainer
            from src.graph.nodes.classify import CATEGORIES

            model_path = "models/improved_model/best_model"
            tok_path = "models/improved_model/best_tokenizer"

            if os.path.exists(model_path) and os.path.exists(tok_path):
                tokenizer = AutoTokenizer.from_pretrained(tok_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                model.eval()
                _EXPLAINER = SequenceClassificationExplainer(model, tokenizer)
        except Exception as e:
            print(f"[Explainer Warning] Could not load explainer: {e}")
            _EXPLAINER = None
    return _EXPLAINER


def extract_influential_tokens(
    title: str,
    text: str,
    target_labels: Optional[List[str]] = None,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    """
    Extracts top influential tokens for predicted macro-categories using Integrated Gradients:
    Returns list of {label, positive_tokens: [{token, weight}], negative_tokens: [{token, weight}]}.
    """
    results: List[Dict[str, Any]] = []
    if not target_labels:
        return results

    explainer = get_explainer()
    if explainer is None:
        # Heuristic fallback if PyTorch explainer is not loaded
        words = [w.strip(".,!?;:\"").lower() for w in f"{title} {text}".split() if len(w) > 4][:top_k]
        for lbl in target_labels[:2]:
            results.append({
                "label": lbl,
                "top_tokens": [{"token": w, "weight": 0.25} for w in words[:4]],
                "method": "heuristic_salience",
            })
        return results

    input_text = f"{title}. {' '.join(text.split()[:120])}"

    for lbl in target_labels[:2]:
        try:
            attributions = explainer(input_text, class_name=lbl)
            # Filter special tokens
            clean_attrs = [
                (t, float(w)) for t, w in attributions
                if t not in ("<s>", "</s>", "<pad>", "[CLS]", "[SEP]") and len(t.strip()) > 1
            ]
            clean_attrs.sort(key=lambda x: abs(x[1]), reverse=True)
            top_toks = [{"token": t, "weight": round(w, 4)} for t, w in clean_attrs[:top_k]]

            results.append({
                "label": lbl,
                "top_tokens": top_toks,
                "method": "integrated_gradients",
            })
        except Exception as e:
            print(f"[Explainer Warning] Failed attribution for {lbl}: {e}")

    return results

