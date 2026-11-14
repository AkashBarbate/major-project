import os
import pickle
import numpy as np
from typing import List, Optional
from src.graph.state import NewsLensState, CredibilitySignalDict
from language_feature import extract_linguistic_features, get_feature_names

_REGRESSOR = None
CRED_MODEL_PATH = "models/credibility_regressor.pkl"

if os.path.exists(CRED_MODEL_PATH):
    try:
        with open(CRED_MODEL_PATH, "rb") as f:
            _REGRESSOR = pickle.load(f)
    except Exception as e:
        print(f"[Credibility Node Warning] Could not load regressor: {e}")


def credibility_node(state: NewsLensState) -> dict:
    """
    Credibility & manipulation analysis node:
    - Extracts 8 transparent linguistic signals
    - Predicts credibility benchmark score (if regressor loaded)
    - Signals are presented as decision-support indicators, not absolute truth judgements.
    """
    clean_title = state.get("clean_title", "")
    clean_text = state.get("clean_text", "")
    url = state.get("url") or ""

    signals: List[CredibilitySignalDict] = []
    score: Optional[float] = None
    label: Optional[str] = None

    try:
        ling_vec = extract_linguistic_features(clean_title, clean_text, url)
        feature_names = get_feature_names()

        INVERTED = {
            'Subjectivity', 'ALL-CAPS ratio', 'Exclamation density',
            'Clickbait signals', 'Sensational vocabulary', 'Headline questions',
        }

        for name, val in zip(feature_names, ling_vec):
            raw_v = float(val)
            disp_v = float(1.0 - raw_v) if name in INVERTED else raw_v
            signals.append({
                "name": name,
                "value": round(raw_v, 4),
                "display_value": round(disp_v, 4),
            })

        if _REGRESSOR is not None:
            pred_val = _REGRESSOR.predict(ling_vec.reshape(1, -1))[0]
            score = round(float(np.clip(pred_val, 0, 100)), 1)

            if score >= 75:
                label = "High linguistic reliability"
            elif score >= 50:
                label = "Moderate linguistic reliability"
            else:
                label = "Manipulative / low reliability cues"
    except Exception as e:
        print(f"[Credibility Node Error] {e}")

    steps = list(state.get("execution_steps", []))
    steps.append("credibility_analysis")

    return {
        "credibility_score": score,
        "credibility_label": label,
        "credibility_signals": signals,
        "execution_steps": steps,
    }

