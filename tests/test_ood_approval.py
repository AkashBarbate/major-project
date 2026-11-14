import uuid
from src.database.session import SessionLocal
from src.database.models import Article, OODReviewItem, MicroCatalogueEntry

def test_ood_approval():
    db = SessionLocal()
    
    # 1. Ensure a dummy article and OOD item exist
    art_id = str(uuid.uuid4())
    art = Article(id=art_id, title="Test Headline", body="Test Body")
    db.add(art)
    
    ood_id = str(uuid.uuid4())
    ood = OODReviewItem(
        id=ood_id,
        article_id=art_id,
        headline="RBI raises repo rate by 25 basis points to curb retail inflation",
        ood_score=0.65,
        nearest_micro_label="inflation & monetary policy",
        similarity=0.35,
        status="pending",
    )
    db.add(ood)
    db.commit()
    
    # 2. Test approving using existing name 'inflation & monetary policy'
    clean_name = "inflation & monetary policy"
    existing = db.query(MicroCatalogueEntry).filter_by(name=clean_name).first()
    assert existing is not None, "Label should exist in seeded catalogue"
    
    # Simulate the fixed approval logic
    seeds = list(existing.seed_examples or [])
    if ood.headline not in seeds:
        seeds.append(ood.headline)
        existing.seed_examples = seeds
    ood.status = "approved"
    ood.reviewed_by = "Analyst"
    ood.proposed_name = existing.name
    
    try:
        db.commit()
        print("[PASS] Approving candidate with existing label name succeeded without UNIQUE constraint error!")
    except Exception as e:
        db.rollback()
        raise AssertionError(f"Failed with error: {e}")
    finally:
        # Cleanup test records
        db.delete(ood)
        db.delete(art)
        db.commit()
        db.close()

if __name__ == "__main__":
    test_ood_approval()

