from fastapi import APIRouter, HTTPException
from src.api.schemas import ClassifyOnlyRequest, MacroLabelScore, MicroLabelScore, TopicStatusResponse
from src.graph.nodes.classify import classify_node, CATEGORIES, OPTUNA_THRESHOLDS
from src.graph.nodes.micro_retrieval import micro_retrieval_node
from src.graph.nodes.ingest import clean_text

router = APIRouter(prefix="", tags=["Classification & Taxonomy"])


@router.get("/categories", summary="List supported macro categories and tuned thresholds")
def get_categories():
    """Returns the 8 frozen macro categories with their validation-tuned thresholds."""
    return {
        "categories": [
            {
                "name": cat,
                "threshold": round(float(OPTUNA_THRESHOLDS[i]), 4),
            }
            for i, cat in enumerate(CATEGORIES)
        ]
    }


@router.post("/classify", summary="Fast multi-label macro and micro classification")
def classify_text(req: ClassifyOnlyRequest):
    """
    Lightweight classification endpoint:
    - Runs calibrated RoBERTa macro classifier
    - Retrieves top semantic micro-labels
    - Returns OOD score without running claims or live evidence search.
    """
    if not req.text or len(req.text.strip()) < 15:
        raise HTTPException(status_code=422, detail="Text too short for classification.")

    clean_t = clean_text(req.title)
    clean_b = clean_text(req.text)

    state = {
        "clean_title": clean_t,
        "clean_text": clean_b,
        "raw_title": req.title,
        "raw_text": req.text,
        "execution_steps": [],
    }

    # Run macro classifier
    macro_res = classify_node(state)
    state.update(macro_res)

    # Run micro-label retriever
    micro_res = micro_retrieval_node(state)

    predicted_macros = [
        m["label"] for m in state.get("macro_predictions", []) if m["predicted"]
    ]

    return {
        "macro_labels": state.get("macro_predictions", []),
        "predicted_macro_labels": predicted_macros,
        "overall_confidence": state.get("macro_confidence", 0.5),
        "is_uncertain": state.get("is_macro_uncertain", False),
        "micro_labels": micro_res.get("micro_labels", []),
        "topic_status": {
            "ood_score": micro_res.get("ood_score", 0.0),
            "review_required": micro_res.get("ood_review_required", False),
            "proposed_cluster": micro_res.get("proposed_cluster"),
        },
    }

