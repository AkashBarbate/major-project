from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from src.database.session import get_db
from src.database.models import HumanAnnotation
from src.api.schemas import HumanAnnotationRequest

router = APIRouter(prefix="/annotations", tags=["Human Annotation & Agreement"])


@router.post("", summary="Submit human annotation judgement")
def submit_annotation(req: HumanAnnotationRequest, db: Session = Depends(get_db)):
    """
    Records an independent or adjudicated human annotation for:
    - Article macro/micro labels
    - Check-worthiness decisions
    - Claim-evidence stance judgements (supports, refutes, neutral)
    - Final verdicts.
    """
    ann = HumanAnnotation(
        entity_type=req.entity_type,
        entity_id=req.entity_id,
        annotator=req.annotator,
        label=req.label,
        confidence=req.confidence,
        notes=req.notes or "",
    )
    db.add(ann)
    db.commit()
    db.refresh(ann)

    return {
        "status": "saved",
        "id": ann.id,
        "annotator": ann.annotator,
        "entity_type": ann.entity_type,
        "label": ann.label,
    }


@router.get("/export", summary="Export annotations for agreement evaluation")
def export_annotations(entity_type: str = None, db: Session = Depends(get_db)):
    """Exports human annotations for computing Cohen's kappa and multi-label F1 agreement."""
    query = db.query(HumanAnnotation)
    if entity_type:
        query = query.filter_by(entity_type=entity_type)
    records = query.all()

    return {
        "total_annotations": len(records),
        "annotations": [
            {
                "id": r.id,
                "entity_type": r.entity_type,
                "entity_id": r.entity_id,
                "annotator": r.annotator,
                "label": r.label,
                "confidence": r.confidence,
                "notes": r.notes,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ],
    }

