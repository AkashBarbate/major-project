import uuid
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from src.database.session import get_db
from src.database.models import MicroCatalogueEntry, OODReviewItem
from src.api.schemas import TaxonomyActionRequest, OODReviewActionRequest
from src.taxonomy.llm_suggester import suggest_micro_label_with_llm


router = APIRouter(prefix="/taxonomy", tags=["Taxonomy Governance"])


class LLMSuggestRequest(BaseModel):
    headline: str
    body: str = ""
    nearest_category: str = ""


@router.post("/suggest-label", summary="Use LangChain Gemini to suggest micro-label for OOD article")
def llm_suggest_label(req: LLMSuggestRequest):
    """
    LLM-assisted topic discovery:
    Analyzes an OOD article and generates canonical name, definition, macro category, and aliases.
    """
    return suggest_micro_label_with_llm(
        headline=req.headline,
        body=req.body,
        nearest_category=req.nearest_category,
    )


@router.get("", summary="Get approved micro-label catalogue")
def get_taxonomy(db: Session = Depends(get_db)):
    """Returns the full approved micro-label catalogue grouped by macro category."""
    entries = db.query(MicroCatalogueEntry).all()
    grouped = {}
    for e in entries:
        cat = e.macro_category
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append({
            "id": e.id,
            "name": e.name,
            "definition": e.definition,
            "aliases": e.aliases or [],
            "seed_examples": e.seed_examples or [],
            "status": e.status,
            "version": e.version,
        })
    return {
        "total_labels": len(entries),
        "macro_categories_count": len(grouped),
        "catalogue": grouped,
    }


@router.post("", summary="Add a new approved micro-label")
def create_micro_label(req: TaxonomyActionRequest, db: Session = Depends(get_db)):
    """Human-in-the-loop taxonomy governance: approve and add a new micro-label."""
    existing = db.query(MicroCatalogueEntry).filter_by(name=req.name.strip().lower()).first()
    if existing:
        raise HTTPException(status_code=400, detail="Micro-label with this name already exists.")

    new_id = f"ml_{uuid.uuid4().hex[:8]}"
    entry = MicroCatalogueEntry(
        id=new_id,
        name=req.name.strip().lower(),
        macro_category=req.macro_category,
        definition=req.definition,
        aliases=req.aliases,
        seed_examples=req.seed_examples,
        status="active",
        version=1,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    return {
        "status": "created",
        "id": entry.id,
        "name": entry.name,
        "macro_category": entry.macro_category,
    }


@router.get("/ood-queue", summary="List pending OOD novel topic cluster candidates")
def get_ood_queue(status: str = "pending", db: Session = Depends(get_db)):
    """Lists novel topic candidates awaiting human review and approval."""
    items = db.query(OODReviewItem).filter_by(status=status).order_by(OODReviewItem.created_at.desc()).all()
    return [
        {
            "id": item.id,
            "article_id": item.article_id,
            "headline": item.headline,
            "ood_score": item.ood_score,
            "nearest_micro_label": item.nearest_micro_label,
            "similarity": item.similarity,
            "status": item.status,
            "created_at": item.created_at.isoformat() if item.created_at else None,
        }
        for item in items
    ]


@router.post("/ood-queue/{item_id}/review", summary="Review and approve/reject an OOD cluster candidate")
def review_ood_item(item_id: str, req: OODReviewActionRequest, db: Session = Depends(get_db)):
    """
    Human-in-the-loop review gate:
    Approve an OOD candidate as an accepted new micro-label or reject it.
    """
    item = db.query(OODReviewItem).filter_by(id=item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="OOD queue item not found")

    if req.action == "approve":
        if not req.approved_name or not req.macro_category:
            raise HTTPException(status_code=422, detail="approved_name and macro_category required for approval.")

        clean_name = req.approved_name.strip().lower()
        existing = db.query(MicroCatalogueEntry).filter_by(name=clean_name).first()

        if existing:
            # Label already in catalogue -> merge seed examples safely
            seeds = list(existing.seed_examples or [])
            if item.headline not in seeds:
                seeds.append(item.headline)
                existing.seed_examples = seeds
            item.status = "approved"
            item.reviewed_by = req.reviewer or "Analyst"
            item.proposed_name = existing.name
            db.commit()
            return {
                "status": "merged",
                "message": f"Candidate merged into existing micro-label '{existing.name}'.",
                "micro_label_id": existing.id,
            }
        else:
            # Genuinely novel micro-label -> insert
            new_entry = MicroCatalogueEntry(
                id=f"ml_{uuid.uuid4().hex[:8]}",
                name=clean_name,
                macro_category=req.macro_category,
                definition=f"Discovered novel topic from article: {item.headline}",
                aliases=[clean_name],
                seed_examples=[item.headline],
                status="active",
                version=1,
            )
            db.add(new_entry)
            item.status = "approved"
            item.reviewed_by = req.reviewer or "Analyst"
            item.proposed_name = clean_name
            db.commit()

            return {
                "status": "approved",
                "message": f"Cluster approved and added to catalogue as '{clean_name}'.",
                "micro_label_id": new_entry.id,
            }
    elif req.action == "reject":
        item.status = "rejected"
        item.reviewed_by = req.reviewer or "Analyst"
        db.commit()
        return {
            "status": "rejected",
            "message": "Candidate rejected.",
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid action. Must be 'approve' or 'reject'.")

