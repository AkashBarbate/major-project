import json
import os
from .session import SessionLocal, init_db
from .models import MicroCatalogueEntry


def seed_micro_catalogue(json_path: str = None):
    """Seed the micro-label catalogue from JSON into SQLite."""
    init_db()

    if json_path is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        json_path = os.path.join(root_dir, "data", "catalogue", "micro_labels.json")

    if not os.path.exists(json_path):
        print(f"[Seed] Catalogue file not found at: {json_path}")
        return 0

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = data.get("labels", [])
    db = SessionLocal()
    inserted_count = 0
    updated_count = 0

    try:
        for item in labels:
            existing = db.query(MicroCatalogueEntry).filter_by(id=item["id"]).first()
            if existing:
                existing.name = item["name"]
                existing.macro_category = item["macro_category"]
                existing.definition = item["definition"]
                existing.aliases = item.get("aliases", [])
                existing.seed_examples = item.get("seed_examples", [])
                existing.status = item.get("status", "active")
                updated_count += 1
            else:
                entry = MicroCatalogueEntry(
                    id=item["id"],
                    name=item["name"],
                    macro_category=item["macro_category"],
                    definition=item["definition"],
                    aliases=item.get("aliases", []),
                    seed_examples=item.get("seed_examples", []),
                    status=item.get("status", "active"),
                    version=1,
                )
                db.add(entry)
                inserted_count += 1
        db.commit()
        print(f"[Seed] Successfully seeded {inserted_count} new and updated {updated_count} micro-labels.")
        return inserted_count + updated_count
    except Exception as e:
        db.rollback()
        print(f"[Seed Error] Failed to seed catalogue: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_micro_catalogue()

