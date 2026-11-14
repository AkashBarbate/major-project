"""
NewsLens 3.0 — Inter-Annotator Agreement Evaluation
Computes Cohen's Kappa and percent agreement across annotators (e.g., Akash and Varun)
for check-worthiness, macro/micro categorization, and stance/verdict assignments.
"""

import json
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np
from sqlalchemy.orm import Session
from src.database.session import SessionLocal
from src.database.models import HumanAnnotation, Claim


def compute_cohens_kappa(r1: List[str], r2: List[str]) -> Dict[str, float]:
    """
    Computes Cohen's Kappa statistic between two raters.
    kappa = (P_observed - P_expected) / (1 - P_expected)
    """
    assert len(r1) == len(r2), "Both raters must have the same number of ratings."
    n = len(r1)
    if n == 0:
        return {"kappa": 0.0, "observed_agreement": 0.0, "expected_agreement": 0.0, "n": 0}

    categories = sorted(list(set(r1) | set(r2)))
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    k = len(categories)

    # Confusion matrix
    matrix = np.zeros((k, k), dtype=int)
    for a, b in zip(r1, r2):
        matrix[cat_to_idx[a], cat_to_idx[b]] += 1

    # Observed agreement
    p_obs = np.trace(matrix) / n

    # Expected agreement by chance
    row_sums = matrix.sum(axis=1) / n
    col_sums = matrix.sum(axis=0) / n
    p_exp = np.sum(row_sums * col_sums)

    if p_exp == 1.0:
        kappa = 1.0
    else:
        kappa = (p_obs - p_exp) / (1.0 - p_exp)

    return {
        "kappa": round(float(kappa), 4),
        "observed_agreement": round(float(p_obs), 4),
        "expected_agreement": round(float(p_exp), 4),
        "n_samples": n,
        "categories": categories,
    }


def interpret_kappa(kappa: float) -> str:
    """Landis and Koch (1977) scale for interpreting agreement strength."""
    if kappa < 0.0:
        return "Poor (Disagreement)"
    elif kappa <= 0.20:
        return "Slight Agreement"
    elif kappa <= 0.40:
        return "Fair Agreement"
    elif kappa <= 0.60:
        return "Moderate Agreement"
    elif kappa <= 0.80:
        return "Substantial Agreement"
    else:
        return "Almost Perfect Agreement"


def evaluate_dataset_annotations(db: Optional[Session] = None) -> Dict[str, Any]:
    """
    Fetches dual annotations from SQLite database and calculates agreement metrics.
    """
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True

    try:
        annotations = db.query(HumanAnnotation).all()
        # Group by target_id (claim_id or article_id)
        grouped = defaultdict(dict)
        for ann in annotations:
            grouped[ann.target_id][ann.annotator_name] = ann

        # Find overlapping items evaluated by at least two annotators
        overlap_items = [item for item in grouped.values() if len(item) >= 2]
        
        if not overlap_items:
            return {
                "status": "empty",
                "message": "No overlapping annotations found between annotators yet.",
                "total_annotations": len(annotations),
                "unique_targets": len(grouped)
            }

        annotator_names = sorted(list({ann.annotator_name for ann in annotations}))
        a1, a2 = annotator_names[0], annotator_names[1]

        r1_verdicts, r2_verdicts = [], []
        r1_stances, r2_stances = [], []

        for item in overlap_items:
            if a1 in item and a2 in item:
                ann1, ann2 = item[a1], item[a2]
                if ann1.gold_verdict and ann2.gold_verdict:
                    r1_verdicts.append(ann1.gold_verdict)
                    r2_verdicts.append(ann2.gold_verdict)
                if ann1.gold_stance and ann2.gold_stance:
                    r1_stances.append(ann1.gold_stance)
                    r2_stances.append(ann2.gold_stance)

        results = {
            "annotators": [a1, a2],
            "total_overlapping_items": len(overlap_items),
        }

        if r1_verdicts:
            verdict_stats = compute_cohens_kappa(r1_verdicts, r2_verdicts)
            verdict_stats["interpretation"] = interpret_kappa(verdict_stats["kappa"])
            results["verdict_agreement"] = verdict_stats

        if r1_stances:
            stance_stats = compute_cohens_kappa(r1_stances, r2_stances)
            stance_stats["interpretation"] = interpret_kappa(stance_stats["kappa"])
            results["stance_agreement"] = stance_stats

        return results

    finally:
        if close_db:
            db.close()


if __name__ == "__main__":
    print("[NewsLens 3.0] Running Inter-Annotator Agreement Analysis...")
    report = evaluate_dataset_annotations()
    print(json.dumps(report, indent=2))

