from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "macro_categories" in data
    print("[PASS] GET /health passed")


def test_categories():
    res = client.get("/categories")
    assert res.status_code == 200
    data = res.json()
    assert len(data["categories"]) == 8
    print("[PASS] GET /categories passed (8 categories)")


def test_taxonomy():
    res = client.get("/taxonomy")
    assert res.status_code == 200
    data = res.json()
    assert data["total_labels"] >= 40
    print(f"[PASS] GET /taxonomy passed ({data['total_labels']} labels)")


def test_claims_extraction():
    res = client.post(
        "/claims/extract",
        json={
            "title": "Government announces space mission allocation",
            "text": "The government on Thursday approved an outlay of 12,000 crore rupees for deep space satellite launches. The first mission will lift off in November 2026.",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert len(data["claims"]) > 0
    print(f"[PASS] POST /claims/extract passed ({len(data['claims'])} claims extracted)")


if __name__ == "__main__":
    test_health()
    test_categories()
    test_taxonomy()
    test_claims_extraction()
    print("\nAll Core API router integration tests passed successfully!")
