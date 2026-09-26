"""Score one claim and explain it in words an investigation-desk employee can act on."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from .features import AUTO_APPROVE_LIMIT, FEATURES, ROOT, load_reference, single_features

GOODWILL_RS = 380

REQUIRED = {
    "partner_id": str, "sku": str, "submitted_at": str, "days_since_purchase": (int, float),
    "claim_amount_inr": (int, float), "photo_attached": str, "partner_inspected": str,
    "customer_prior_claims": (int, float),
}


class Scorer:
    def __init__(self):
        art = joblib.load(ROOT / "model" / "model.joblib")
        self.models = art["models"]
        self.cal = art["calibrator"]
        self.threshold = art["threshold"]
        self.trained_until = art["trained_until"]
        self.partners, self.products = load_reference()
        self.snapshot = pd.read_csv(ROOT / "model" / "partner_snapshot.csv", index_col="partner_id")

    # ------------------------------------------------------------ validation
    def validate(self, rec: dict) -> list[str]:
        errs = []
        for k, typ in REQUIRED.items():
            if k not in rec or rec[k] in (None, ""):
                errs.append(f"missing field '{k}'")
            elif not isinstance(rec[k], typ):
                errs.append(f"'{k}' has the wrong type")
        if errs:
            return errs
        if rec["partner_id"] not in set(self.partners.partner_id):
            errs.append(f"unknown partner_id '{rec['partner_id']}' (not in partners.csv)")
        if rec["sku"] not in set(self.products.sku):
            errs.append(f"unknown sku '{rec['sku']}' (not in products.csv)")
        if str(rec["photo_attached"]).upper() not in ("Y", "N"):
            errs.append("photo_attached must be Y or N")
        if str(rec["partner_inspected"]).upper() not in ("Y", "N"):
            errs.append("partner_inspected must be Y or N")
        if rec["claim_amount_inr"] <= 0:
            errs.append("claim_amount_inr must be positive")
        try:
            pd.Timestamp(rec["submitted_at"])
        except Exception:
            errs.append("submitted_at is not a date/time (use YYYY-MM-DD HH:MM, IST)")
        return errs

    # ------------------------------------------------------------ scoring
    def score(self, rec: dict) -> dict:
        rec = dict(rec)
        rec["photo_attached"] = str(rec["photo_attached"]).upper()
        rec["partner_inspected"] = str(rec["partner_inspected"]).upper()
        X = single_features(rec, self.partners, self.products, self.snapshot)
        raw = float(np.mean([m.predict_proba(X[FEATURES])[:, 1] for m in self.models]))
        p = float(self.cal.predict_proba(np.log([[raw + 1e-6]]))[0, 1])
        contrib = np.mean([m.predict(X[FEATURES], pred_contrib=True)[0][:-1] for m in self.models], axis=0)
        amt = float(rec["claim_amount_inr"])
        ev = p * amt - (1 - p) * GOODWILL_RS          # rupees gained by checking this claim
        in_queue = p >= self.threshold
        if in_queue and ev > 0:
            action = "SEND TO INVESTIGATION DESK"
        elif in_queue:
            action = "HIGH RISK BUT LOW VALUE: pay, and watch the partner"
        else:
            action = "PAY: no review needed"
        return {
            "fraud_probability": round(p, 4),
            "risk_level": "high" if in_queue else ("medium" if p >= self.threshold / 3 else "low"),
            "recommended_action": action,
            "expected_value_of_checking_inr": round(ev),
            "reasons": self._reasons(X.iloc[0], contrib, rec),
            "context": {
                "queue_threshold": round(self.threshold, 4),
                "desk_capacity": "40 claims / month (ops-policy §5)",
                "model_trained_on_claims_until": self.trained_until,
            },
        }

    def _reasons(self, x, contrib, rec) -> list[dict]:
        """Top drivers in plain English. Direction comes from the model's own
        per-feature contribution (SHAP), wording from the claim's actual values."""
        text = {
            "p_fraud_rate_sm": lambda: (
                f"Partner {rec['partner_id']} has no track record yet (no decided claims)"
                if x.p_past_labelled == 0 else
                f"Partner {rec['partner_id']} has {int(x.p_past_fraud)} confirmed fraud(s) in "
                f"{int(x.p_past_labelled)} decided claims" if x.p_past_fraud > 0 else
                f"Partner {rec['partner_id']} has no confirmed fraud in {int(x.p_past_labelled)} decided claims"),
            "p_past_fraud": lambda: (
                f"Partner {rec['partner_id']} has {int(x.p_past_fraud)} confirmed fraud(s) on record"
                if x.p_past_fraud > 0 else
                f"Partner {rec['partner_id']} has no track record yet (no decided claims)"
                if x.p_past_labelled == 0 else f"Partner {rec['partner_id']} has no fraud on record"),
            "band_1900_1999": lambda: (
                f"Amount Rs {x.amount:,.0f} is just under the Rs {AUTO_APPROVE_LIMIT:,.0f} auto-approval limit"
                if x.band_1900_1999 else f"Amount Rs {x.amount:,.0f} is not in the Rs 1,900-1,999 band"),
            "auto_approved": lambda: ("Claim is auto-approved with no inspection (under Rs 2,000, after 1 May 2026)"
                                      if x.auto_approved else "Claim needs inspection before payment"),
            "not_inspected": lambda: ("No inspection sign-off" if x.not_inspected else "Inspection sign-off present"),
            "no_photo": lambda: ("No photo attached" if x.no_photo else "Photo attached"),
            "prior_claims": lambda: f"Customer has {int(x.prior_claims)} earlier warranty claim(s)",
            "tenure_days": lambda: f"Partner onboarded {int(x.tenure_days)} days before this claim",
            "new_partner": lambda: ("Partner is under a year old" if x.new_partner else "Partner is over a year old"),
            "p_band_share": lambda: f"{x.p_band_share:.0%} of this partner's past claims sit just under Rs 2,000",
            "p_noinsp_share": lambda: f"{x.p_noinsp_share:.0%} of this partner's past claims had no inspection",
            "p_prior_mean": lambda: f"This partner's customers average {x.p_prior_mean:.1f} earlier claims",
            "p_past_n": lambda: f"Partner has submitted {int(x.p_past_n)} earlier claims",
            "amount": lambda: f"Claim amount Rs {x.amount:,.0f}",
            "amt_ratio": lambda: f"Claim is {x.amt_ratio:.0%} of the product's list price",
            "days": lambda: f"Claim made {int(x.days)} days after purchase",
            "warranty_used": lambda: f"{x.warranty_used:.0%} of the warranty period used",
            "post_may": lambda: "Submitted under the post-May auto-approval rules",
            "type_franchise": lambda: "Partner is a franchise" if x.type_franchise else "Partner is not a franchise",
            "type_freelance_technician": lambda: ("Partner is a freelance technician" if x.type_freelance_technician
                                                  else "Partner is not a freelance technician"),
            "p_past_labelled": lambda: (f"{int(x.p_past_labelled)} of this partner's claims have been decided"
                                        if x.p_past_labelled else "None of this partner's claims have been decided yet"),
        }
        # 'post_may' is the same for every claim scored today, so it explains nothing to
        # a reader; it still counts in the score, just isn't listed.
        hide = {"post_may"}
        order = [i for i in np.argsort(-np.abs(contrib)) if FEATURES[i] not in hide]
        out = []
        for i in order[:4]:
            f = FEATURES[i]
            if abs(contrib[i]) < 0.05:
                continue
            out.append({"reason": text[f](), "effect": "raises risk" if contrib[i] > 0 else "lowers risk"})
        return out or [{"reason": "Nothing about this claim stands out from normal claims", "effect": "neutral"}]
