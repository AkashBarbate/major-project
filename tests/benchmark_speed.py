import time
from src.graph.pipeline import newslens_pipeline


def benchmark_pipeline_speed():
    sample_article = {
        "raw_title": "Reserve Bank of India raises repo rate by 25 basis points to 6.75 percent",
        "raw_text": (
            "The Reserve Bank of India on Friday announced an increase in the benchmark lending repo rate "
            "by 25 basis points to 6.75 percent to curb rising retail inflation. Central bank Governor Shaktikanta Das "
            "stated that retail inflation rose by 5.4 percent in the previous quarter, exceeding the medium-term target band. "
            "The monetary policy committee voted five to one in favor of the rate hike, citing persistent core price pressures. "
            "Commercial banks are expected to transmit the policy rate increase to retail borrowing rates shortly."
        ),
        "url": None,
    }

    print("=" * 60)
    print("NewsLens 3.0 — Prediction Latency Benchmark")
    print("=" * 60)

    # Run 1: Warmup & Initial Execution
    t0 = time.time()
    res1 = newslens_pipeline.invoke(sample_article)
    warmup_time = time.time() - t0
    print(f"\n[Run 1 — Initial Warmup Execution]: {warmup_time:.2f} seconds")

    # Run 2: Active Production Execution (Models in RAM, Embeddings Cached)
    t1 = time.time()
    res2 = newslens_pipeline.invoke(sample_article)
    production_time = time.time() - t1
    print(f"[Run 2 — In-Memory Production Run]: {production_time:.2f} seconds")

    print("\n--- Predictions & Evidence ---")
    predicted_macros = [m["label"] for m in res2.get("macro_predictions", []) if m["predicted"]]
    print(f"Predicted Macro Labels: {predicted_macros}")
    print(f"Micro Labels: {[m['name'] for m in res2.get('micro_labels', [])]}")
    print(f"Extracted Claims: {len(res2.get('claims', []))}")
    for c in res2.get("claims", []):
        print(f"  * Claim: '{c['text'][:60]}...' -> Verdict: {c.get('verdict')} (Conf: {c.get('confidence')})")

    print(f"\nLatency Reduction: Production run completed in {production_time:.2f}s!")
    print("=" * 60)


if __name__ == "__main__":
    benchmark_pipeline_speed()

