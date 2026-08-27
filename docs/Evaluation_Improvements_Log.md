# ShopSense — Evaluation Methodology: Before → After

This documents how the evaluation numbers changed over the course of this
project, and specifically *why* — three distinct bugs, one design decision,
and one important dataset finding, each changing the numbers for a
different, identifiable reason. Kept separate from the main README (which
covers the earlier CF pipeline bugs) and the SASRec notes (which cover the
NaN bug) — this one is specifically about how to trust the *metric itself*.

## The core lesson

A recall number is only as trustworthy as the evaluation code that produced
it. Three separate times in this project, a plausible-looking number turned
out to be measuring a bug, not the model. Documenting each one because
recognizing the *shape* of a suspicious metric (a discontinuous jump, two
models producing identical output, a headline number too good to be true)
is a transferable skill — the specific dataset won't repeat, but the
pattern-recognition will.

## Round 1 — The leakage-inflated split (root cause, fixed first)

**Symptom:** ~98% "leakage rate" reported, Item-Item CF Recall@50 jumping
from ~1% to ~39%.

**Cause:** the train/test split excluded only the exact matching row of
the held-out transaction — events *after* the cutoff stayed in training.

**Fix:** per-user chronological cutoff with an assertion. See the main
README for full detail; this round is the foundation everything below
builds on.

## Round 2 — The `filter_already_liked_items` padding bug

**Symptom:**

```
Item-Item CF:  Recall@5=0.0070  Recall@10=0.0090  Recall@20=0.0133  Recall@50=0.4987  Recall@100=0.6873
```

A 37x jump between Recall@20 and Recall@50, with no equivalent jump in the
Popularity baseline evaluated the same way.

**Cause:** `implicit`'s `filter_already_liked_items=True` does not actually
remove already-seen items from the returned list — it zeroes their score
and leaves them as filler once genuine candidates run out. Requesting
`N=100` for a sparse user (most of them) triggered this padding well before
50 real candidates existed. Since ~98% of held-out targets are items the
user already interacted with, that padding tail frequently *contained the
correct answer* — producing a metric that looked like model skill but was
actually measuring how the padding happened to be constructed.

**Fix:** evaluate with `exclude_seen=False`, matching the actual task
(next-interaction prediction, where a previously-seen item is a legitimate
answer) instead of relying on a flag that didn't do what its name promised.

**Result after fixing this alone:**

```
Item-Item CF:  Recall@5=0.850  Recall@10=0.890  Recall@20=0.918  Recall@50=0.941  Recall@100=0.975
```

Smooth curve, no cliff — but see Round 3 for why this number *still*
wasn't the full story.

## Round 3 — The trivial-baseline check (the important one)

**Symptom:** nothing crashed and no curve looked broken this time — the
issue only shows up if you specifically ask "is this recall coming from
real model skill?"

**The check:**

```python
def repeat_own_history_recommend(user_idx, N):
    row = user_item_matrix[user_idx]
    order = np.argsort(-row.data)
    return list(row.indices[order][:N])
```

A baseline with *zero* machine learning — just re-sort the user's own
training history by interaction weight.

**Result:**

```
Trivial repeat-own-history:  Recall@5=0.962  Recall@10=0.975  Recall@20=0.980  Recall@50=0.984  Recall@100=0.986
```

**This beats every real model, including Item-Item CF, at every K.**

**What this means:** on this dataset, "next interaction" is dominated by
repeat behavior (view something 3 times, then buy it). Item-Item CF's
85–97% recall wasn't demonstrating collaborative intelligence — it was
mostly succeeding at the same thing a zero-effort lookup already does
better. This is a genuine, important finding about the dataset, not a code
bug, and it directly explains why SASRec's re-ranking (which weighs session
*context*, not just recency) sometimes scored *lower* than plain CF at low
K — it was disrupting a trivial "recommend what they just looked at" echo
that the raw CF score was riding on.

## Full before → after comparison

| Metric | Broken (Round 2 bug) | Fixed, full test set | Trivial baseline |
|---|---|---|---|
| Recall@5 | 0.0070 | 0.850 | **0.962** |
| Recall@10 | 0.0090 | 0.890 | **0.975** |
| Recall@20 | 0.0133 | 0.918 | **0.980** |
| Recall@50 | 0.4987 | 0.941 | **0.984** |
| Recall@100 | 0.6873 | 0.975 | **0.986** |

Neither the "broken" nor the "fixed, full test set" number is the one to
put on a slide unqualified — the middle column is honestly computed, but
it's measuring a task (repeat-interaction prediction) that a non-ML lookup
already solves. The number that actually demonstrates the model's value is
the **discovery-mode metric** (`exclude_seen=True`, scoped only to test
users whose target was genuinely novel) — that's the case where a trivial
history lookup has literally nothing to offer, and a real model has to earn
its score.

## What changed in the final architecture as a result

- **Item-Item CF and SASRec still ship** — but positioned correctly, as the
  engine behind a "Recommended for You" surface, not as the source of the
  headline recall number.
- **A separate "Recently Viewed" rail** (direct history lookup, no model)
  is the honest way to serve the repeat-interaction case this dataset is
  dominated by — cheaper and more reliable than asking a trained model to
  approximate something a lookup already does better.
- **User-User CF and ALS are not shipped.** Neither ever demonstrated a
  clear advantage over Item-Item CF in any evaluation round, broken or
  fixed.
- **The discovery-mode metric is the one reported as "model performance"**
  in the final results — not the full-test-set number, which is dominated
  by a pattern no model is needed to catch.

## Why this is worth keeping, unedited

Every round here changed the *metric*, not the underlying model — the
actual `ItemItemRecommender` and `SASRec` objects never changed between
rounds 2 and 3. That's the whole point of documenting it: the model was
never the problem, the measurement was, three separate times, in three
separate ways. That's a more useful thing to be able to explain in an
interview than a single clean number ever would be.
