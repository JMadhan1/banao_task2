"""Loading, cleaning and feature engineering for Kestrel warranty claims.

One feature definition is shared by batch scoring (train / backtest / predictions.csv)
and the single-record API, so the service can never drift from what was validated.
Every partner-history feature uses ONLY claims submitted strictly before the claim
being scored (no look-ahead).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"

AUTO_APPROVE_FROM = pd.Timestamp("2026-05-01")  # ops-policy §5
AUTO_APPROVE_LIMIT = 2000.0                      # ops-policy §5
BASE_RATE = 0.0127                               # labelled fraud rate after dedupe
PRIOR_N = 10                                     # smoothing weight for partner fraud rate
BEH_N = 5                                        # smoothing weight for behavioural shares

FEATURES = [
    "amount", "amt_ratio", "band_1900_1999", "post_may", "auto_approved",
    "not_inspected", "no_photo", "prior_claims", "days", "warranty_used",
    "tenure_days", "new_partner", "type_franchise", "type_freelance_technician",
    "p_past_n", "p_past_labelled", "p_past_fraud", "p_fraud_rate_sm",
    "p_band_share", "p_noinsp_share", "p_prior_mean",
]


# ---------------------------------------------------------------- loading
def load_reference():
    partners = pd.read_csv(DATA / "partners.csv")
    products = pd.read_csv(DATA / "products.csv")
    return partners, products


def load_all() -> pd.DataFrame:
    """train + test in one time-ordered frame, cleaned.

    Cleaning decisions (see evidence/DATA_ISSUES.md):
      * resubmitted claims (same claim_id twice, identical except submitted_at) ->
        keep the first submission;
      * blank is_fraud = undecided -> kept for partner *behaviour* features but never
        used as a label;
      * product_serial is not used (partner-typed, and after normalising it collides
        across unrelated SKUs, so it carries no reliable identity).
    """
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test_unlabelled.csv")
    tr = tr.sort_values("submitted_at").drop_duplicates("claim_id", keep="first")
    al = pd.concat([tr.assign(split="train"), te.assign(split="test")], ignore_index=True)
    partners, products = load_reference()
    al = al.merge(partners, on="partner_id", how="left").merge(products, on="sku", how="left")
    al["t"] = pd.to_datetime(al["submitted_at"])
    return al.sort_values(["t", "claim_id"]).reset_index(drop=True)


# ---------------------------------------------------------------- per-claim features
def claim_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that only need the claim itself + reference tables."""
    X = pd.DataFrame(index=df.index)
    a = df["claim_amount_inr"].astype(float)
    t = pd.to_datetime(df["submitted_at"])
    X["amount"] = a
    X["amt_ratio"] = a / df["list_price_inr"]
    X["band_1900_1999"] = ((a >= 1900) & (a < AUTO_APPROVE_LIMIT)).astype(int)
    X["post_may"] = (t >= AUTO_APPROVE_FROM).astype(int)
    X["auto_approved"] = ((a < AUTO_APPROVE_LIMIT) & (X["post_may"] == 1)).astype(int)
    X["not_inspected"] = (df["partner_inspected"].str.upper() == "N").astype(int)
    X["no_photo"] = (df["photo_attached"].str.upper() == "N").astype(int)
    X["prior_claims"] = df["customer_prior_claims"].astype(float)
    X["days"] = df["days_since_purchase"].astype(float)
    X["warranty_used"] = X["days"] / (df["warranty_months"] * 30.4)
    X["tenure_days"] = (t - pd.to_datetime(df["onboarded_date"])).dt.days.astype(float)
    X["new_partner"] = (X["tenure_days"] < 365).astype(int)
    X["type_franchise"] = (df["partner_type"] == "franchise").astype(int)
    X["type_freelance_technician"] = (df["partner_type"] == "freelance_technician").astype(int)
    return X


def _history_from_sums(X, n, labelled, fraud, band, noinsp, prior):
    X["p_past_n"] = n
    X["p_past_labelled"] = labelled
    X["p_past_fraud"] = fraud
    X["p_fraud_rate_sm"] = (fraud + BASE_RATE * PRIOR_N) / (labelled + PRIOR_N)
    X["p_band_share"] = (band + 0.03 * BEH_N) / (n + BEH_N)
    X["p_noinsp_share"] = (noinsp + 0.3 * BEH_N) / (n + BEH_N)
    X["p_prior_mean"] = (prior + 0.3 * BEH_N) / (n + BEH_N)
    return X


def batch_features(al: pd.DataFrame, label_mask: pd.Series | None = None) -> pd.DataFrame:
    """Features for a time-ordered frame. Partner history = claims strictly earlier.

    label_mask: which rows' is_fraud may be used as history (default: train rows with a
    decided label). Backtests pass a mask that hides labels at/after the cut-off.
    """
    X = claim_features(al)
    g = al["partner_id"]
    if label_mask is None:
        label_mask = (al["split"] == "train") & al["is_fraud"].notna()
    known = label_mask.astype(float)
    fr = al["is_fraud"].where(label_mask, 0).fillna(0).astype(float)

    def past(s):
        return s.groupby(g).cumsum() - s

    n = al.groupby("partner_id").cumcount().astype(float)
    X = _history_from_sums(
        X, n, past(known), past(fr),
        past(X["band_1900_1999"].astype(float)),
        past(X["not_inspected"].astype(float)),
        past(X["prior_claims"].astype(float)),
    )
    return X[FEATURES]


def partner_snapshot(al: pd.DataFrame) -> pd.DataFrame:
    """Per-partner running totals over ALL claims in `al` (used by the API)."""
    X = claim_features(al)
    lab = (al["split"] == "train") & al["is_fraud"].notna()
    d = pd.DataFrame({
        "partner_id": al["partner_id"],
        "n": 1.0,
        "labelled": lab.astype(float),
        "fraud": al["is_fraud"].where(lab, 0).fillna(0).astype(float),
        "band": X["band_1900_1999"].astype(float),
        "noinsp": X["not_inspected"].astype(float),
        "prior": X["prior_claims"].astype(float),
    })
    return d.groupby("partner_id").sum()


def single_features(record: dict, partners: pd.DataFrame, products: pd.DataFrame,
                    snapshot: pd.DataFrame) -> pd.DataFrame:
    """Features for one API record, using the partner snapshot as its history."""
    df = pd.DataFrame([record])
    df = df.merge(partners, on="partner_id", how="left").merge(products, on="sku", how="left")
    X = claim_features(df)
    pid = record["partner_id"]
    if pid in snapshot.index:
        s = snapshot.loc[pid]
        vals = (s["n"], s["labelled"], s["fraud"], s["band"], s["noinsp"], s["prior"])
    else:
        vals = (0.0,) * 6
    X = _history_from_sums(X, *[pd.Series([v], index=X.index) for v in vals])
    return X[FEATURES]
