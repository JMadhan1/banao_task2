# Evidence: does it work, and how often does it not?

All numbers come from `python -m src.train`, which writes `backtest_monthly.csv`,
`metrics.json`, `errors_missed_fraud.csv` and `errors_false_alarms.csv` in this folder.
Rerunning it reproduces them.

## How it was tested

The test is a **walk-forward backtest that copies how the model would really be used.** For
each month from Jan to Jun 2026:
1. Hide every label from that month onward.
2. Train on the earlier months only.
3. Score that month's claims.
4. Send the top 40 to the "desk", which is the ops-policy §5 capacity.
5. Count the frauds caught, the genuine customers held (₹380 goodwill each, §4), and the
   rupees stopped.

There is no random split. Random splits leak future partners' fraud into the past and look
far better than reality.

Each month is compared with two simple alternatives:
- **New-partner rule:** Ritu's hypothesis. Flag claims from partners under a year old, the
  largest first.
- **Random:** check 40 claims at random.

## Results

| Month | Claims | Frauds | Accuracy if we flag nothing | **Model: frauds in top 40** | Model net ₹ | New-partner rule: frauds | Random: frauds |
|---|---|---|---|---|---|---|---|
| Jan 26 | 730 | 8 | 98.9% | **5** | +19,993 | 0 | 0 |
| Feb 26 | 694 | 5 | 99.3% | **5** | +36,393 | 0 | 0 |
| Mar 26 | 738 | 9 | 98.8% | **6** | +29,364 | 0 | 0 |
| Apr 26 | 709 | 9 | 98.7% | **3** | +3,000 | 0 | 1 |
| May 26 ⚠ | 709 | 14 | 98.0% | **0** | −15,200 | 6 | 1 |
| **Jun 26** | 713 | 22 | 96.9% | **16** | **+12,991** | 4 | 4 |

"Net ₹" is fraud stopped minus ₹380 goodwill for each genuine claim held.

**June is the month that looks like the test period**, because the model had already seen
one month of the new auto-approval rules. In June:
- **16 of the 40 checks were fraud**, so 40% of flagged claims were fraud. Checking at
  random finds 3%.
- It caught 16 of the month's 22 frauds, which is **73% recall**.
- PR-AUC was **0.47** (90% bootstrap interval 0.31–0.64). ROC-AUC was **0.90** (0.83–0.96).
- **₹22,111 of fraud stopped, or ₹553 per check** (Farhan's measure). After goodwill, that's
  **₹12,991 net**, or ₹325 per check.
- The new-partner rule caught 4 and lost ₹5,717. Random caught 4 and lost ₹9,092.

**May is the failure case, and it's kept in on purpose.** In the first month of
auto-approval, a model trained only on older data caught **0 of 14** frauds, and its AUC of
0.25 is worse than a coin flip. The fraud pattern flipped: large, inspected claims from older
partners gave way to small, uninspected claims from a few new partners. **Any rule change
Kestrel makes will break the model the same way until it is retrained on the new pattern.**

## How often it is wrong, and on what kind of case (June, top 40)

- **False alarms: 24 of 40 (60%).** 20 of the 24 were uninspected claims. 15 came from
  partners that already had confirmed fraud, e.g. 3 from SP3129 and 2 each from SP3095 and
  SP3318. These are genuine claims from suspicious partners. Each costs ₹380 in goodwill.
- **Missed fraud: 6 of 22 (27%).** 5 of the 6 came from partners with **no confirmed fraud
  yet**:
  - SP3118 and SP3286 were onboarded on 3–4 June and had their first frauds that same month.
  - SP3275 had a ₹17,866 inspected claim.
  - SP3121 had a small claim.

  The model leans heavily on a partner's track record, so **a brand-new fraudulent partner
  goes unflagged until its first fraud is confirmed.** This is its main weakness. The test
  period has 11 partners onboarded after the data ends (49 claims) that face exactly this
  problem.
- **Before May** (Jan–Apr), the model caught 19 of 31 frauds. It never filled all 40 slots
  with fraud, because there were only 5–9 frauds a month.

## What score I expect on the hidden test (July–September 2026, 2,252 claims)

- **Metric: PR-AUC (average precision).** The submission is a ranking score and fraud is
  1–4% of claims. PR-AUC rewards putting fraud at the top, which is what the 40-a-month desk
  needs. ROC-AUC flatters rare-event models, and accuracy means nothing here.
- **Expected PR-AUC: about 0.35, likely range 0.20–0.50. Expected ROC-AUC: about 0.85.**
  - June scored 0.47, and I'm expecting less for two reasons:
    1. The test runs 1–3 months past the training data, and the drift from month to month
       has been large.
    2. The test includes new partners with no history, which is the error mode above.
  - Holding it up: the 7 partners behind the post-May fraud are now known, and 87 of the
    top 120 test claims come from them.
- **Expected frauds in the test set:** the calibrated scores add up to about 86, around 3.8%
  of claims. That continues the upward trend (May 2.0%, June 3.1%).

## Automated checks (`python -m pytest -q`, 8 tests)

- The API returns a probability, an action and reasons.
- A risky claim scores more than 10× a routine one.
- Bad input returns a readable 422, not a crash.
- A partner with no history still gets scored.
- `predictions.csv` matches `sample_submission.csv` row for row.
- Partner history never uses a claim from the same moment or later.
- Resubmitted claims are removed.

## Limits of this evidence

- Only **22 frauds in June**, so the figures move a lot from month to month. The bootstrap
  interval above is honest about that.
- The backtest assumes a month's labels are known by the start of the next month. If
  investigations take longer, partner history is staler and results will be somewhat worse.
- Labels only show what the desk confirmed (see DATA_ISSUES #7). Fraud nobody investigated
  is counted as genuine, which flatters recall.
