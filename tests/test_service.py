"""python -m pytest -q   (from task2/, after `python -m src.train`)"""
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.server import app
from src.features import FEATURES, ROOT, batch_features, load_all

client = TestClient(app)
RISKY = {"partner_id": "SP3160", "sku": "KH-WP-02", "submitted_at": "2026-09-20 14:30",
         "claim_amount_inr": 1975, "days_since_purchase": 180, "customer_prior_claims": 2,
         "photo_attached": "N", "partner_inspected": "N"}
ROUTINE = {"partner_id": "SP3265", "sku": "KH-MG-02", "submitted_at": "2026-09-20 11:05",
           "claim_amount_inr": 1240, "days_since_purchase": 410, "customer_prior_claims": 0,
           "photo_attached": "Y", "partner_inspected": "Y"}


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_score_returns_probability_action_and_reasons():
    r = client.post("/score", json=RISKY).json()
    assert 0 <= r["fraud_probability"] <= 1
    assert r["recommended_action"] and r["reasons"]
    assert all({"reason", "effect"} <= set(x) for x in r["reasons"])


def test_risky_claim_scores_above_routine_claim():
    hi = client.post("/score", json=RISKY).json()["fraud_probability"]
    lo = client.post("/score", json=ROUTINE).json()["fraud_probability"]
    assert hi > 10 * lo


def test_bad_input_is_rejected_politely():
    r = client.post("/score", json={"partner_id": "NOPE"})
    assert r.status_code == 422 and "details" in r.json()
    r = client.post("/score", json={**RISKY, "partner_id": "SP9999"})
    assert r.status_code == 422 and "unknown partner_id" in r.json()["details"][0]


def test_unknown_future_partner_still_scores():
    # a partner in partners.csv with no claims in the snapshot must not crash
    snap = pd.read_csv(ROOT / "model" / "partner_snapshot.csv").partner_id
    p = pd.read_csv(ROOT / "data" / "raw" / "partners.csv").partner_id
    unseen = sorted(set(p) - set(snap))
    if unseen:
        r = client.post("/score", json={**ROUTINE, "partner_id": unseen[0]})
        assert r.status_code == 200


def test_predictions_file_matches_sample_submission():
    pred = pd.read_csv(ROOT / "predictions.csv")
    sample = pd.read_csv(ROOT / "data" / "raw" / "sample_submission.csv")
    assert list(pred.columns) == ["claim_id", "score"]
    assert list(pred.claim_id) == list(sample.claim_id)
    assert pred.score.between(0, 1).all() and pred.score.nunique() > 100


def test_partner_history_never_looks_ahead():
    al = load_all()
    X = batch_features(al)
    first = al.groupby("partner_id").head(1).index
    assert (X.loc[first, ["p_past_n", "p_past_labelled", "p_past_fraud"]] == 0).all().all()
    # test rows' own (unknown) labels can never feed history
    assert not al.loc[al.split == "test", "is_fraud"].notna().any()
    assert list(X.columns) == FEATURES


def test_resubmitted_claims_are_deduplicated():
    al = load_all()
    assert al.claim_id.is_unique
