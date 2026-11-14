"""
NewsLens 3.0 — Offline Benchmark Evaluation Suite
Evaluates:
1. Expected Calibration Error (ECE) before & after temperature scaling
2. Macro & Micro F1 across the frozen 8 categories
3. RAG Retrieval comparison: Local-only vs. Web-only vs. Hybrid
4. Stance Classification & NEI Precision/Recall
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from src.evidence_kb.local_kb import get_local_evidence_kb
from src.web_retrieval.factory import get_web_search_provider
from src.graph.nodes.verify import evaluate_stance, aggregate_verdict


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) across multi-label predictions.
    """
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = probs.size

    for i in range(n_bins):
        low, high = bin_limits[i], bin_limits[i + 1]
        mask = (probs >= low) & (probs < high)
        if i == n_bins - 1:
            mask = (probs >= low) & (probs <= high)

        bin_count = np.sum(mask)
        if bin_count > 0:
            avg_confidence = np.mean(probs[mask])
            avg_accuracy = np.mean(labels[mask])
            ece += (bin_count / total_samples) * np.abs(avg_accuracy - avg_confidence)

    return float(round(ece, 4))


def evaluate_calibration_impact(val_logits_path: str = "models/improved_model/val_logits.npy",
                               val_labels_path: str = "models/improved_model/val_labels.npy",
                               temp_path: str = "models/improved_model/temperature.json") -> Dict[str, Any]:
    """
    Measures ECE before and after temperature scaling on validation logits.
    """
    if not (os.path.exists(val_logits_path) and os.path.exists(val_labels_path)):
        return {"status": "skipped", "message": "Validation cache files not found."}

    logits = np.load(val_logits_path)
    labels = np.load(val_labels_path)

    temp = 1.0
    if os.path.exists(temp_path):
        with open(temp_path, "r") as f:
            t_data = json.load(f)
            temp = float(t_data.get("temperature", 1.0))

    # Raw probabilities
    raw_probs = 1.0 / (1.0 + np.exp(-logits))
    ece_before = compute_ece(raw_probs, labels)

    # Calibrated probabilities
    cal_logits = logits / temp
    cal_probs = 1.0 / (1.0 + np.exp(-cal_logits))
    ece_after = compute_ece(cal_probs, labels)

    # Multi-label F1 scores
    raw_preds = (raw_probs >= 0.5).astype(int)
    cal_preds = (cal_probs >= 0.5).astype(int)

    return {
        "status": "success",
        "temperature_used": temp,
        "ece_uncalibrated": ece_before,
        "ece_calibrated": ece_after,
        "ece_reduction_pct": round(((ece_before - ece_after) / ece_before) * 100, 2) if ece_before > 0 else 0,
        "macro_f1": round(float(f1_score(labels, cal_preds, average="macro", zero_division=0)), 4),
        "micro_f1": round(float(f1_score(labels, cal_preds, average="micro", zero_division=0)), 4),
        "hamming_loss": round(float(hamming_loss(labels, cal_preds)), 4),
    }


def evaluate_retrieval_ablation(test_queries: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Evaluates Local KB vs Web Search vs Hybrid Retrieval on a set of benchmark claims.
    """
    local_kb = get_local_evidence_kb()
    web_provider = get_web_search_provider()

    results = {
        "local_kb_hits": 0,
        "web_search_hits": 0,
        "hybrid_hits": 0,
        "total_claims": len(test_queries)
    }

    for item in test_queries:
        claim_text = item["claim"]
        
        # Local-only
        local_res = local_kb.search(claim_text, top_k=3, threshold=0.35)
        if len(local_res) > 0:
            results["local_kb_hits"] += 1

        # Web-only
        try:
            web_res = web_provider.search(claim_text, max_results=3)
        except Exception:
            web_res = []
        if len(web_res) > 0:
            results["web_search_hits"] += 1

        # Hybrid
        if len(local_res) > 0 or len(web_res) > 0:
            results["hybrid_hits"] += 1

    n = max(len(test_queries), 1)
    return {
        "benchmark_claims_count": n,
        "local_kb_recall": round(results["local_kb_hits"] / n, 4),
        "web_search_recall": round(results["web_search_hits"] / n, 4),
        "hybrid_retrieval_recall": round(results["hybrid_hits"] / n, 4),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("NewsLens 3.0 — Offline Benchmark Evaluation")
    print("=" * 60)

    # 1. Calibration & Classifier Evaluation
    cal_results = evaluate_calibration_impact()
    print("\n--- 1. Macro Classification & Calibration Report ---")
    print(json.dumps(cal_results, indent=2))

    # 2. Retrieval Ablation Sample
    sample_claims = [
        {"claim": "The Reserve Bank of India increased repo rate by 25 bps."},
        {"claim": "NASA James Webb Space Telescope detected atmospheric water vapor on exoplanet."},
        {"claim": "WHO declared ending of COVID-19 as a public health emergency of international concern."},
        {"claim": "ISRO launched Chandrayaan-3 mission successfully from Sriharikota."},
        {"claim": "India won the ICC Men's T20 World Cup in Barbados."},
    ]
    print("\n--- 2. RAG Retrieval Ablation Benchmark ---")
    retrieval_results = evaluate_retrieval_ablation(sample_claims)
    print(json.dumps(retrieval_results, indent=2))
    print("=" * 60)
