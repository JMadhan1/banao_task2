# Memo: Warranty fraud flag

**To:** Ritu Deshpande, Head of D2C Operations · **Cc:** Farhan Sheikh, Meenal Joshi
**Re:** What the claims data says, and what to do next week

**The decision:** Put the review queue live next week. It picks the 40 claims a month the
investigation desk should check. **Drop "accuracy above 97%" as the board KPI.** Report
"fraud caught per 40 checks, in rupees" instead.

**Why not accuracy.** Only about 1 claim in 80 is fraud. A system that approves every claim
and catches nothing scores 98.7% "accuracy", so it would pass the board's test while
stopping ₹0. The honest measure is how much of what the desk checks turns out to be fraud.

**The number.** Tested on June as if it were live, the queue sent 40 claims to the desk and
**16 were fraud (40%)**. That was 16 of the month's 22 frauds. Picking 40 claims at random
finds about 1 fraud in 30.

**The rupees.**
- In June, the queue stopped **₹22,100 of fraud, about ₹550 for every claim checked.**
- After ₹380 goodwill for each genuine customer held, it is **about ₹13,000 net a month**,
  roughly ₹1.5 lakh a year.
- Confirmed fraud is only about ₹45,000 a month in total, so no model can save much more
  than that. The bigger lever is the policy change below.

**On the new partners: you were half right.** Since the May auto-approval rule, 33 of the
36 confirmed frauds came from just **7 of the ~60 new partners**. The other new partners had
1 fraud in 155 claims. Meenal is right that most new partners are fine. Flagging all of them
catches a quarter as much fraud and loses money. The 7 are SP3160, SP3232, SP3318, SP3129,
SP3319, SP3118 and SP3286.

**What changed in May.** Since claims under ₹2,000 stopped needing inspection, fraud has
moved to claims priced just under ₹2,000 (₹1,900–1,999) with no inspection. The fraud rate
has gone from about 1% of claims to 2–3%.

**What to do next week**
1. **Review the 7 partners above.** Hold their payouts pending review. Their claims make up
   most of the queue.
2. **Remove auto-approval for partners in their first 6 months** until they have a clean
   record, or at least require a photo.
3. **Go live with the queue:** the top 40 claims a month to the desk, each with the reasons
   shown on screen.
4. **Hold back 5 of the 40 checks for random claims from brand-new partners.** The model
   can't spot a new bad partner until its first fraud is confirmed. It missed two in June
   this way.
5. **Retrain monthly**, and immediately after any rule change. When auto-approval started,
   the old pattern stopped working and the model caught nothing that month.

**The risk to know about.** On average, 6 of every 10 claims the desk checks will be genuine,
costing about ₹9,000 a month in goodwill. That is the price of catching the other 4.
