# Kestrel Home: Warranty Claim Review (Task 2, Variant C)

This tool picks the claims worth sending to Kestrel's investigation desk. The desk can check
at most **40 claims a month** (ops-policy §5). For every new warranty claim, the model returns
a fraud probability, a recommended action and plain-English reasons. It also produces
`predictions.csv` for the hidden test set.

Read in this order:
1. [`memo/memo_to_ritu.md`](memo/memo_to_ritu.md): the decision, the number and the rupees, in plain English.
2. [`evidence/EVIDENCE.md`](evidence/EVIDENCE.md): how we know it works, and how often it doesn't.
3. [`evidence/DATA_ISSUES.md`](evidence/DATA_ISSUES.md): what is wrong with the data and what was done about it.
4. [`submission-form.md`](submission-form.md): the filled-in form, including handover notes.

## Folder layout

```
task2/
├── data/raw/                 the client pack, unchanged (train.csv, test_unlabelled.csv, partners.csv,
│                             products.csv, sample_submission.csv, ops-policy.pdf, email-thread.txt, README.txt)
├── src/
│   ├── features.py           cleaning + features (shared by training and the API)
│   ├── train.py              backtest -> final model -> predictions.csv + evidence files
│   └── scoring.py            scores one claim and turns it into reasons a desk employee can read
├── app/
│   ├── server.py             FastAPI: POST /score, GET / (screen), GET /health
│   └── static/index.html     the review screen
├── model/                    trained model + partner-history snapshot (made by src/train.py)
├── evidence/                 backtest results, error lists, write-ups
├── memo/memo_to_ritu.md
├── tests/test_service.py
├── predictions.csv           deliverable 1
└── submission-form.md        deliverable 6
```

## Run it on a clean machine

You need Python 3.10 or newer. **No API key is needed.** The model runs locally, and the
product does not call any paid model API.

```bash
cd task2
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.train          # ~15 s: backtest, trains the model, writes predictions.csv + evidence/
uvicorn app.server:app --port 8000
```

Open http://localhost:8000, press **Example: suspicious**, then **Check claim**.

If you start the server before training, `/health` and `/score` return a polite 503 telling
you to run `python -m src.train` first. They do not crash.

### Call the endpoint directly

```bash
curl -s -X POST localhost:8000/score -H 'Content-Type: application/json' -d '{
  "partner_id": "SP3160", "sku": "KH-WP-02", "submitted_at": "2026-09-20 14:30",
  "claim_amount_inr": 1975, "days_since_purchase": 180, "customer_prior_claims": 2,
  "photo_attached": "N", "partner_inspected": "N"}'
```

```json
{
  "fraud_probability": 0.5163,
  "risk_level": "high",
  "recommended_action": "SEND TO INVESTIGATION DESK",
  "expected_value_of_checking_inr": 836,
  "reasons": [
    {"reason": "Partner SP3160 has 10 confirmed fraud(s) in 39 decided claims", "effect": "raises risk"},
    {"reason": "Customer has 2 earlier warranty claim(s)", "effect": "raises risk"},
    ...
  ]
}
```

Bad input (a missing field, an unknown partner or SKU, a malformed date) returns a 422 that
lists what is wrong.

### Tests

```bash
python -m pytest -q          # 8 tests: API, input validation, predictions shape, no look-ahead, dedupe
```

## How the score works

- **Model:** 5 LightGBM models with different random seeds, averaged. They are trained on
  decided CRM claims. Claims from May 2026 onward count 8× more, because the fraud pattern
  changed when auto-approval started. The raw score is then calibrated to a probability,
  using out-of-time June results.
- **Features (21):**
  - About the claim: amount, the ₹1,900–1,999 band just under the auto-approval limit,
    whether it was auto-approved, inspection sign-off, photo, the customer's earlier claims,
    how much of the warranty had been used, and the partner's type and tenure.
  - The partner's history, using only claims submitted *before* this one: confirmed fraud
    (smoothed), the share of claims in the ₹1,900–1,999 band, the share of uninspected
    claims, and the average number of earlier claims per customer.
- **Not used:** product serial (unreliable), the free-text fault description and inspector
  note (they barely separate fraud from genuine, and are written by the partner being
  checked), and city.
- **Queue:** a claim is flagged when its probability is at or above the score of the 40th
  riskiest claim per month in the test period. The "value of checking" is
  `p × amount − (1 − p) × ₹380 goodwill`.

## Data handling

Ops-policy §10 says Kestrel's data must not be uploaded to public repositories. `data/raw/`
is kept here exactly as received because the submission asks for it. **The repository must
be private** and shared only with the reviewer's address.
