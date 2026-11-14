import json
from src.graph.pipeline import newslens_pipeline

sample_article = {
    "raw_title": "RBI raises repo rate by 25 basis points to curb retail inflation",
    "raw_text": (
        "The Reserve Bank of India on Friday announced an increase in the benchmark "
        "lending repo rate by 25 basis points to 6.75 percent. Central bank Governor "
        "stated that retail inflation rose by 5.4 percent in the previous quarter, exceeding "
        "the medium-term target band. The monetary policy committee voted unanimously to maintain "
        "a hawkish stance on consumer price stability. In contrast, industry bodies argued that "
        "further borrowing cost hikes could dampen private capital expenditure in manufacturing."
    ),
    "url": "https://www.thehindu.com/business/economy/rbi-rate-hike-inflation",
    "publisher": "The Hindu",
}


def test_run():
    print("Executing NewsLens 3.0 LangGraph pipeline...")
    final_state = newslens_pipeline.invoke(sample_article)

    print("\n=== EXECUTION STEPS ===")
    print(" -> ".join(final_state.get("execution_steps", [])))

    print("\n=== MACRO CLASSIFICATION ===")
    for m in final_state.get("macro_predictions", []):
        if m["predicted"] or m["score"] >= 0.3:
            print(f"  - {m['label']}: score={m['score']:.4f} (thresh={m['threshold']}) [PREDICTED: {m['predicted']}]")

    print("\n=== RETRIEVED MICRO LABELS ===")
    for mic in final_state.get("micro_labels", []):
        print(f"  - [{mic['macro_category']}] {mic['name']} (score={mic['score']:.4f})")

    print(f"\n=== OOD STATUS ===")
    print(f"  OOD Score: {final_state.get('ood_score')} | Review Required: {final_state.get('ood_review_required')}")

    print("\n=== EXTRACTED CLAIMS & VERDICTS ===")
    for clm in final_state.get("claims", []):
        print(f"  Claim: \"{clm['text']}\"")
        print(f"  Check-worthiness: {clm['checkworthiness_score']} | Verdict: {clm['verdict']} (conf={clm['confidence']})")
        print(f"  Evidence count: {len(clm.get('evidence', []))}")
        for ev in clm.get("evidence", [])[:2]:
            print(f"    * [{ev['stance']}] {ev['domain']} ({ev['source_group']}): \"{ev['passage'][:80]}...\"")

    print("\n=== SOURCE DIVERSITY SUMMARY ===")
    print(json.dumps(final_state.get("source_diversity_summary", {}), indent=2))

    print("\nPipeline execution test completed successfully!")


if __name__ == "__main__":
    test_run()

