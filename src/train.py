"""Backtest, train the final model, write predictions.csv and the service artefacts.

    python -m src.train          (from task2/)

Outputs
  predictions.csv                       one score per test claim (higher = more likely fraud)
  model/model.joblib                    LightGBM ensemble + calibrator + queue threshold
  model/partner_snapshot.csv            per-partner history the API uses
  evidence/backtest_monthly.csv         month-by-month out-of-time results
  evidence/metrics.json                 headline numbers quoted in the memo / form
  evidence/errors_*.csv                 the claims the model got wrong in the backtest
"""
from __future__ import annotations

import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from .features import FEATURES, ROOT, batch_features, load_all, partner_snapshot

CAPACITY_PER_MONTH = 40      # ops-policy §5
GOODWILL_RS = 380            # ops-policy §4: genuine claim held for review
RECENT_FROM = pd.Timestamp("2026-05-01")
RECENT_WEIGHT = 8.0          # post-May claims count 8x: the fraud pattern changed in May
SEEDS = range(5)
EVID = ROOT / "evidence"
MODEL = ROOT / "model"


def make_model(seed):
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5,
        random_state=seed, verbose=-1,
    )


def training_rows(al, cutoff):
    """Decided labels, CRM only, before cutoff.

    Legacy Zoho rows are excluded from TRAINING because Zoho stored 'undecided' as 0
    (email from Tanmay), so ~3% of their zeros are not real negatives. They are still
    used for partner-history features.
    """
    return (al.split == "train") & al.is_fraud.notna() & (al.source == "crm") & (al.t < cutoff)


def fit_ensemble(X, y, t):
    w = np.where(t >= RECENT_FROM, RECENT_WEIGHT, 1.0)
    return [make_model(s).fit(X, y, sample_weight=w) for s in SEEDS]


def raw_score(models, X):
    return np.mean([m.predict_proba(X[FEATURES])[:, 1] for m in models], axis=0)


def at_capacity(y, s, amt, k):
    o = np.argsort(-s)[:k]
    tp = int(y[o].sum())
    caught = float((amt[o] * y[o]).sum())
    return dict(k=k, tp=tp, fp=k - tp, precision=tp / k, recall=tp / max(y.sum(), 1),
                rs_caught=caught, rs_goodwill=GOODWILL_RS * (k - tp),
                rs_net=caught - GOODWILL_RS * (k - tp))


def backtest(al):
    """For each month M: hide labels from M on, train on < M, score M, pick top 40."""
    rows, preds = [], []
    for m in pd.period_range("2026-01", "2026-06", freq="M"):
        start, end = m.start_time, (m + 1).start_time
        hide = (al.split == "train") & al.is_fraud.notna() & (al.t < start)
        X = batch_features(al, label_mask=hide)
        trm = training_rows(al, start)
        vm = (al.split == "train") & al.is_fraud.notna() & (al.t >= start) & (al.t < end)
        models = fit_ensemble(X[trm], al.is_fraud[trm], al.t[trm])
        s = raw_score(models, X[vm])
        y = al.is_fraud[vm].values.astype(int)
        amt = al.claim_amount_inr[vm].values
        # naive alternatives, for comparison
        newp = X.loc[vm, "new_partner"].values + 1e-6 * rankdata(amt)   # Ritu's hypothesis
        rng = np.random.default_rng(0).random(len(y))
        r = dict(month=str(m), claims=int(vm.sum()), frauds=int(y.sum()),
                 fraud_rs=float((amt * y).sum()),
                 accuracy_if_flag_nothing=1 - y.mean(),
                 auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s))
        k = CAPACITY_PER_MONTH
        for name, sc in [("model", s), ("new_partner_rule", newp), ("random", rng)]:
            for kk, vv in at_capacity(y, sc, amt, k).items():
                r[f"{name}_{kk}"] = vv
        rows.append(r)
        preds.append(al.loc[vm, ["claim_id", "t", "partner_id", "claim_amount_inr",
                                 "partner_inspected", "is_fraud"]].assign(score=s, month=str(m)))
    return pd.DataFrame(rows), pd.concat(preds)


def bootstrap_ci(p, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, s = p.is_fraud.values.astype(int), p.score.values
    aps, aucs = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        aps.append(average_precision_score(y[i], s[i]))
        aucs.append(roc_auc_score(y[i], s[i]))
    return dict(pr_auc_ci=np.percentile(aps, [5, 95]).round(3).tolist(),
                auc_ci=np.percentile(aucs, [5, 95]).round(3).tolist())


def main():
    EVID.mkdir(exist_ok=True)
    MODEL.mkdir(exist_ok=True)
    al = load_all()

    # ---------------- 1. out-of-time backtest
    bt, bp = backtest(al)
    bt.to_csv(EVID / "backtest_monthly.csv", index=False)
    # June = the situation the test is in (model has already seen >=1 month of the
    # post-May auto-approval regime). May = the month of the switch, model has seen none.
    post = bp[bp.month == "2026-06"]
    may = bp[bp.month == "2026-05"]
    pre = bp[bp.month < "2026-05"]

    # ---------------- 2. calibration: map raw scores -> probability using the
    #                     out-of-time post-May backtest scores (the regime the test is in)
    cal = LogisticRegression(C=1.0).fit(np.log(post[["score"]].values + 1e-6), post.is_fraud.astype(int))

    # ---------------- 3. final model on everything we have
    X = batch_features(al)
    trm = training_rows(al, pd.Timestamp("2100-01-01"))
    models = fit_ensemble(X[trm], al.is_fraud[trm], al.t[trm])

    te = al.split == "test"
    s_test = raw_score(models, X[te])
    p_test = cal.predict_proba(np.log(s_test.reshape(-1, 1) + 1e-6))[:, 1]
    pred = pd.DataFrame({"claim_id": al.claim_id[te].values, "score": p_test.round(6)})
    sample = pd.read_csv(ROOT / "data" / "raw" / "sample_submission.csv")
    pred = sample[["claim_id"]].merge(pred, on="claim_id", how="left")
    assert pred.score.notna().all() and len(pred) == len(sample)
    pred.to_csv(ROOT / "predictions.csv", index=False)

    # queue threshold: the score of the 40th-highest claim per month in the test period
    n_months = al.t[te].dt.to_period("M").nunique()
    k = CAPACITY_PER_MONTH * n_months
    threshold = float(np.sort(p_test)[::-1][k - 1])

    snap = partner_snapshot(al)
    snap.to_csv(MODEL / "partner_snapshot.csv")
    joblib.dump(dict(models=models, calibrator=cal, threshold=threshold,
                     features=FEATURES, trained_until=str(al.t[al.split == "train"].max())),
                MODEL / "model.joblib")

    # ---------------- 4. error analysis (post-May backtest, top-40 per month)
    errs = []
    for m, g in post.groupby("month"):
        g = g.sort_values("score", ascending=False).reset_index(drop=True)
        g["flagged"] = g.index < CAPACITY_PER_MONTH
        errs.append(g)
    errs = pd.concat(errs)
    errs[(~errs.flagged) & (errs.is_fraud == 1)].to_csv(EVID / "errors_missed_fraud.csv", index=False)
    errs[(errs.flagged) & (errs.is_fraud == 0)].to_csv(EVID / "errors_false_alarms.csv", index=False)

    # did the missed frauds come from partners with no fraud history at that time?
    Xb = batch_features(al, label_mask=(al.split == "train") & al.is_fraud.notna() & (al.t < "2026-06-01"))
    hist = pd.Series(Xb.p_past_fraud.values, index=al.claim_id)
    missed = errs[(~errs.flagged) & (errs.is_fraud == 1)]
    caught = errs[(errs.flagged) & (errs.is_fraud == 1)]

    post_bt = bt[bt.month == "2026-06"]
    may_bt = bt[bt.month == "2026-05"].iloc[0]
    metrics = dict(
        regime_switch_may=dict(
            note="trained only on pre-May data, scored on May (first month of auto-approval)",
            frauds=int(may_bt.frauds), caught=int(may_bt.model_tp), auc=float(may_bt.auc),
            pr_auc=float(may_bt.pr_auc), rs_net=float(may_bt.model_rs_net),
            new_partner_rule_caught=int(may_bt.new_partner_rule_tp)),
        backtest_june=dict(
            months=post_bt.month.tolist(), claims=int(post_bt.claims.sum()),
            frauds=int(post_bt.frauds.sum()),
            pr_auc=float(average_precision_score(post.is_fraud, post.score)),
            auc=float(roc_auc_score(post.is_fraud, post.score)),
            **bootstrap_ci(post),
            flagged=int(post_bt.model_k.sum()), caught=int(post_bt.model_tp.sum()),
            precision_at_capacity=float(post_bt.model_tp.sum() / post_bt.model_k.sum()),
            recall_at_capacity=float(post_bt.model_tp.sum() / post_bt.frauds.sum()),
            rs_caught=float(post_bt.model_rs_caught.sum()),
            rs_goodwill=float(post_bt.model_rs_goodwill.sum()),
            rs_net=float(post_bt.model_rs_net.sum()),
            fraud_rs_total=float(post_bt.fraud_rs.sum()),
            new_partner_rule_caught=int(post_bt.new_partner_rule_tp.sum()),
            new_partner_rule_rs_net=float(post_bt.new_partner_rule_rs_net.sum()),
            random_caught=int(post_bt.random_tp.sum()),
            accuracy_if_flag_nothing=float(1 - post.is_fraud.mean()),
            missed_with_no_partner_fraud_history=int((hist.reindex(missed.claim_id).values == 0).sum()),
            missed_total=int(len(missed)),
            caught_with_no_partner_fraud_history=int((hist.reindex(caught.claim_id).values == 0).sum()),
        ),
        backtest_pre_may=dict(
            months=bt[bt.month < "2026-05"].month.tolist(),
            pr_auc=float(average_precision_score(pre.is_fraud, pre.score)),
            auc=float(roc_auc_score(pre.is_fraud, pre.score)),
            caught=int(bt[bt.month < "2026-05"].model_tp.sum()),
            frauds=int(bt[bt.month < "2026-05"].frauds.sum()),
        ),
        test=dict(rows=int(te.sum()), months=int(n_months), queue_size=int(k),
                  queue_threshold=threshold, mean_score=float(p_test.mean()),
                  expected_frauds=float(p_test.sum())),
    )
    (EVID / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(bt[["month", "claims", "frauds", "auc", "pr_auc", "model_tp", "model_precision",
              "model_rs_net", "new_partner_rule_tp", "random_tp"]].round(3).to_string())
    print(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    main()
