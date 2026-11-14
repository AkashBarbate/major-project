import os
from src.graph.nodes.micro_retrieval import micro_retrieval_node

def test_multilabel_retrieval():
    # Multi-faceted business article spanning monetary policy, inflation, banking, and interest rates
    state = {
        "article_id": "art_multi_test",
        "clean_title": "RBI raises benchmark repo rate by 25 bps to fight inflation, affecting home loans and commercial banks",
        "clean_text": (
            "The Reserve Bank of India on Friday increased the benchmark repo rate by 25 basis points to 6.75 percent. "
            "Governor Shaktikanta Das highlighted that headline retail inflation and core consumer prices remain elevated above target. "
            "Commercial banks and banking institutions are expected to immediately transmit this monetary policy rate increase "
            "into higher interest rates on home loans, personal lending, and deposit rates across financial markets."
        ),
        "top_macro_labels": ["Business and Economy"],
        "execution_steps": [],
    }

    result = micro_retrieval_node(state)
    labels = result.get("micro_labels", [])

    print("\n" + "=" * 60)
    print(f"Multi-Topic Micro-Label Retrieval Results (Count: {len(labels)}):")
    print("=" * 60)
    for idx, l in enumerate(labels, 1):
        print(f"{idx}. {l['name']} ({l['macro_category']}) — Score: {l['score']:.4f} [LLM Discovered: {l.get('is_llm_discovered', False)}]")

    print(f"\nOOD Score: {result.get('ood_score')} | Review Required: {result.get('ood_review_required')}")
    print("=" * 60)
    assert len(labels) >= 2, "Should return multiple micro-labels for a rich multi-topic article"
    print("[PASS] Multi-label retrieval verified successfully!")


if __name__ == "__main__":
    test_multilabel_retrieval()

