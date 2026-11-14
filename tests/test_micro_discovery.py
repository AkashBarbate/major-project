import os
from src.graph.nodes.micro_retrieval import micro_retrieval_node, load_catalogue, get_embedder
from src.database.session import SessionLocal
from src.database.models import MicroCatalogueEntry


def test_micro_discovery_and_vector_indexing():
    # 1. State for a niche article with 0 pre-existing micro-labels in our catalogue
    state = {
        "article_id": "test_art_001",
        "clean_title": "Ted Turner family puts 20 landscape paintings from Montana ranch up for auction",
        "clean_text": "More than 20 landscape paintings from media mogul Ted Turner's Montana ranch are heading to auction at Christie's in New York, with an estimated value of over $20 million. The collection features rare 19th-century works depicting Western American landscapes, wildlife, and historical scenes.",
        "top_macro_labels": ["Entertainment"],
        "execution_steps": [],
    }

    # 2. Run micro_retrieval_node
    result = micro_retrieval_node(state)
    labels = result.get("micro_labels", [])
    print(f"\nRetrieved / Discovered micro-labels ({len(labels)}):")
    for l in labels:
        print(f"  - {l['name']} ({l['macro_category']}) [Score: {l['score']}] [LLM Discovered: {l.get('is_llm_discovered', False)}]")

    assert len(labels) > 0, "Should have retrieved or discovered micro-labels"
    
    # 3. Verify in SQLite database
    db = SessionLocal()
    try:
        discovered = [l for l in labels if l.get("is_llm_discovered")]
        if discovered:
            first_disc = discovered[0]["name"]
            db_entry = db.query(MicroCatalogueEntry).filter_by(name=first_disc).first()
            assert db_entry is not None, f"Discovered label '{first_disc}' should be persisted in SQLite"
            print(f"[PASS] Persisted in SQLite: {db_entry.name} -> {db_entry.macro_category}")
    finally:
        db.close()

    print("[PASS] Micro-label discovery and vector DB indexing test successful!")


if __name__ == "__main__":
    test_micro_discovery_and_vector_indexing()
