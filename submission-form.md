# Submission form: Kestrel Home Warranty Claim Review (Task 2 V3, Variant C)

## What did you build, and what business outcome does it move? State the number and the money.

I built a review queue that picks the 40 claims a month the investigation desk should check.
The 40 is the desk's capacity (ops-policy §5). The queue has three parts:
- a model;
- one API endpoint that returns a fraud probability, an action and plain-English reasons;
- one screen that calls it.

The outcome it moves is **fraud stopped per check, in rupees**, which is the measure Farhan
asked for. It is not accuracy.
- In the June walk-forward backtest, **16 of the 40 checks were fraud (40%)**, against 3% at
  random. That was 16 of June's 22 frauds, or 73% recall.
- **₹22,111 of fraud stopped, or ₹553 per check.**
- After ₹380 goodwill for each genuine customer held, **₹12,991 net a month, about ₹1.5 lakh
  a year.**
- Ritu's "new partners" rule, run through the same test, caught 4 and lost ₹5,717.

I'm stating the ceiling honestly: all confirmed fraud comes to about ₹45k a month. The larger
saving is the policy fix in the memo (no auto-approval for partners in their first 6 months),
not the model.

## What score do you expect predictions.csv to get on the hidden outcomes, on which metric, and why that metric? Say how you estimated it.

**PR-AUC (average precision): about 0.35, likely range 0.20–0.50. ROC-AUC: about 0.85.**

- **Why PR-AUC:** the file is a ranking and fraud is 1–4% of claims. PR-AUC rewards putting
  fraud at the top, which is exactly what a 40-check desk needs.
  - Accuracy is meaningless here: flagging nothing scores 98.7%.
  - ROC-AUC looks flattering on rare events.
- **How I estimated it:** a walk-forward backtest. For each month, I trained only on earlier
  months and scored that month.
  - June is the month most like the test period (one month of the new auto-approval rules
    already seen). It scored PR-AUC 0.47 (bootstrap 90% interval 0.31–0.64) and ROC-AUC
    0.90 (0.83–0.96).
  - I expect less on the test for two reasons:
    1. The test runs 1–3 months past the training data, and the drift from month to month
       has been large (May scored 0.01 because of the rule change).
    2. 11 partners onboarded in Jul–Aug have no history, and history is the model's
       strongest signal.
- The calibrated scores add up to about 86 expected frauds in 2,252 claims (3.8%).

## How do you know it works? Sample size, how you checked, error rate, and the kind of case it gets wrong.

**Walk-forward backtest, January to June 2026.** Each month: 690–740 claims and 5–22 frauds.
Train on earlier months, send the top 40 to the "desk", count the frauds and rupees. Each
month is compared with the new-partner rule and with random picks. Details are in
`evidence/EVIDENCE.md`, and the raw rows are in `evidence/errors_*.csv`.

**June: 713 claims, 22 frauds.**
- **False-alarm rate: 60%** (24 of 40 checks were genuine).
- **Miss rate: 27%** (6 of 22).

**The kind of case it gets wrong:**
- **Misses:** a brand-new fraudulent partner before its first fraud is confirmed. 5 of the 6
  misses were like this (SP3118 and SP3286, onboarded 3–4 June), plus one ₹17,866 inspected
  claim from an older partner.
- **False alarms:** genuine, uninspected claims from partners that already have confirmed
  fraud (15 of 24).
- **Rule changes:** in May, the first month of auto-approval, the model trained on older data
  caught 0 of 14. It fails whenever the rules change, until it is retrained.

There are also 8 automated tests (`pytest`) covering the API, input validation, the
prediction file format, no look-ahead in the features, and deduplication.

## Did you change, narrow, or push back on the client's ask? What, when, and why.

1. **Pushed back on "accuracy above 97%".** I did this at the very start, before any
   modelling. Fraud is 1.27% of claims, so approving everything scores 98.7% and stops ₹0.
   I replaced it with fraud stopped per 40 checks, in rupees. That is Farhan's framing, and it
   matches the desk's real constraint.
2. **Narrowed "flag fraudulent claims before we pay them" to "rank claims for a desk that can
   check 40 a month".** I did this after reading ops-policy §5. A yes/no flag that flags 200
   claims can't be acted on.
3. **Tested Ritu's hypothesis rather than building it in.** "New partners are the problem" is
   true of only 7 of about 60 new partners. The other new partners had 1 fraud in 155 claims.
   Partner age is one feature among 21, not a rule, and the memo names the 7 partners.
4. **Recommended a policy change alongside the model.** Remove auto-approval for partners in
   their first 6 months. The model can't catch a new bad partner until its first fraud is
   confirmed, and the policy closes that gap.

## What is wrong with what you are handing us, or with the data we handed you?

**Data** (full table in `evidence/DATA_ISSUES.md`):
- **681 resubmitted claims:** the same `claim_id` twice, differing only in `submitted_at`. I
  kept the first.
- **Legacy Zoho rows store "undecided" as 0**, so about 150 legacy zeros aren't real
  negatives. I didn't train on legacy rows.
- **215 blank (undecided) CRM labels.** I dropped them as labels.
- **`product_serial` is unusable.** It's in mixed formats, and after normalising, 3,745
  serials repeat across different SKUs.
- **Labels mostly mean "paid, not investigated".** There are 11,348 decided labels but only
  about 600 desk checks. So fraud nobody looked at is labelled genuine, and recall is
  overstated.
- **The partner's inspection sign-off didn't stop fraud.** Before May, 95 of 105 frauds had
  one.

**My deliverable:**
- The API's partner history is a **snapshot as of 30 June**. It doesn't update from new
  claims until someone reruns `python -m src.train`.
- The **backtest assumes investigation outcomes arrive within a month.**
- **Calibration rests on one month** (June, 22 frauds).
- The **8× weight on post-May claims** and the smoothing constants were chosen on the June
  backtest, the same month I report. That makes the June figures somewhat optimistic, which
  is why my test estimate is lower.
- **Free text isn't used at all.** There may be signal there that I left on the table.

## What did you deliberately leave out, and why that rather than something else?

- **An LLM inside the product.** The decision is driven by partner history and claim
  structure, not language. A model API would add cost and a key dependency, and wouldn't earn
  its place. The reasons come from the model's own per-feature contributions, so they are
  never made up.
- **Free-text features.** Fault descriptions and inspector notes are written by the partner
  under suspicion. Fraud rates by phrase were all 0.5–1.8%, with no signal.
- **Product serials and city:** unreliable (serials), and no signal beyond partner (city).
- **Hyper-parameter search and model zoo.** There are only 22 frauds in the key month, so
  tuning would fit noise. I kept one sensible LightGBM setup with 5 seeds averaged, and
  compared it with a simple rule and logistic regression during research.
- **Docker, auth, a database.** "A small thing that runs beats a large thing that does not."

## Anything you built or found that nobody asked for?

- **The May regime flip.** Before May, fraud was large, inspected claims from a set of 7
  *older* partners. After May, it's claims priced just under ₹2,000, uninspected, from 7
  *new* partners. The old set went quiet: 3 frauds in Q2 2026. That's why a model trained on
  history fails at every rule change, and why I recommend monthly retraining.
- **"Value of checking" per claim** (p × amount − (1 − p) × ₹380). Some high-risk claims are
  worth less to check than they cost in goodwill, and the screen says so.
- **Holding back some checks for random new-partner claims** (memo, step 4), to generate
  labels where the model is blind.

## What did you use AI for?

- **Claude (Claude Opus 5.5, in the Claude app, working in a cloud workspace):**
  - **Research:** profiling the data; finding the duplicates, the regime change, the partner
    concentration and the label problems.
  - **Code:** features, backtest, API, screen, tests.
  - **First drafts** of the evidence, memo and this form.

  I checked each number against the script output.
- **Where it helped:** fast data profiling across many hypotheses, and building the
  walk-forward backtest correctly (no look-ahead) on the first attempt.
- **Where it wasted time:** the first backtest pooled May and June together. That buried the
  June result and made calibration poor, so I had to split them. A service restart command
  also killed its own shell once.
- **What I threw away:**
  - a random train/test split (it leaks future partner fraud);
  - logistic regression, and a hand-weighted rule score (both weaker than LightGBM in the June
    backtest);
  - training on legacy rows (contaminated zeros);
  - a pooled May–June calibration;
  - a "post-May" reason line in the UI (it's the same for every current claim, so it explains
    nothing).
- **Cost:** the Claude subscription I already pay for. No paid API calls.

**Screen recording (≤ 3 min):** _add link here_

## Your Public Google Drive Link

_add link here_

## Someone picks this up on Monday and you are unreachable. The three things they need to know.

1. **Run `python -m src.train` whenever new investigation outcomes come in, at least monthly.**
   It retrains the model, refreshes the partner-history snapshot the API uses, reruns the
   backtest and rewrites `predictions.csv`. Check `evidence/backtest_monthly.csv`: if the
   latest month's `model_tp` falls well below June's 16, the fraud pattern has moved.
2. **The model breaks when Kestrel changes a rule.** It caught nothing in May 2026, the first
   month of auto-approval. After any change to the approval policy, hold back extra checks
   for random claims until a month of new outcomes is in.
3. **The biggest blind spot is a new partner that commits fraud before anything is on
   record.** Keep 5 of the 40 monthly checks for random claims from brand-new partners. Also
   push the policy change (no auto-approval in a partner's first 6 months). The model alone
   can't close this gap.

## Honest hours spent.

_add number here_

## Github Repo Link

https://github.com/JMadhan1/banao_task (the `task2/` folder). The repo must be **private**
and shared with the invitation address, because ops-policy §10 forbids publishing Kestrel
data.

## What does one prediction cost, and what would a month cost at Kestrel's volume (about 750 warranty claims a month)?

**No paid calls. ₹0 in API spend.** The model is a local LightGBM ensemble.

- **One prediction:** about 25 ms including the HTTP call (measured: median 26 ms over all 2,252 test claims), so effectively ₹0.
- **A month:** 750 claims × ₹0 = **₹0 in API cost.** Monthly retraining takes about 15
  seconds of CPU.
- **Hosting**, if it runs as a service: the smallest cloud VM (1 vCPU) is more than enough.
  That's roughly ₹500–800 a month if Kestrel doesn't already have a server to run it on.
- **The real cost is the goodwill from false alarms:** about 24 genuine claims held a month
  × ₹380 = **about ₹9,100 a month**. That's already counted in the ₹12,991 net figure above.
