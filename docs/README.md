# ShopSense — Collaborative Filtering Recommendation Engine

An item-based collaborative filtering recommender built on the RetailRocket
e-commerce dataset (2.76M events, ~1.1M users, ~210K items), with a
leakage-free chronological evaluation and a saved, reloadable production
model.

## What it does

Given a visitor ID, recommends the top-N products they're most likely to
interact with next, based on item-item collaborative filtering:

```
recommend(172, N=10)
→ Visitor: 172
  1. Item 119736  (score=8.42, source=item_item_cf)
  2. Item 213834  (score=6.10, source=item_item_cf)
  ...
```

Falls back to a popularity-ranked list for cold-start visitors with no
prior history.

## Problems I ran into, and how I solved them

This project went through several real debugging rounds. Documenting them
here rather than hiding them, since diagnosing and fixing each one is most
of the actual engineering work.

### 1. Data leakage in the train/test split (~98% leakage rate)

**Symptom:** a check comparing test targets against training history showed
98.25% of test items already appeared in that user's "training" data —
suspiciously high, and Item-Item Recall@50 jumped from ~1% to ~39% between
K=20 and K=50, which is not a normal recall curve shape.

**Root cause:** the train/test split excluded only the exact
`(visitorid, itemid, timestamp)` row matching each held-out transaction.
Every *other* event for that user — including events that happened
*after* the held-out transaction — stayed in the training set. The model
was training on the future.

**Fix:** switched to a per-user cutoff timestamp. For each user, the cutoff
is the timestamp of their last transaction; training data for that user is
everything strictly before the cutoff, full stop. Added an assertion
(`train_events['timestamp'] < cutoff` for every test user) that fails
immediately if this is ever broken again, instead of surfacing as a
suspicious number three cells later.

One thing this was *not*: a user viewing an item before buying it (e.g.
visitor 172 viewing item 10034, then buying it) is not leakage — that's the
real browsing-before-buying signal, and it happens before the cutoff. The
task is "predict what they'll interact with next," and prior views of the
eventual purchase are exactly the kind of signal that should be in
training.

### 2. ALS `KeyError` from inconsistent matrix orientation

**Symptom:** `KeyError: 223332` when converting recommended indices back to
item IDs — the model was returning indices that didn't exist in the item
mapping.

**Root cause:** `implicit`'s `AlternatingLeastSquares.fit()` orientation
convention (whether it expects a user×item or item×user matrix) isn't
consistent across library versions, and I'd flip-flopped between the two
across different cells without re-verifying, so the fitted factors didn't
line up with the ID mappings anymore.

**Fix:** instead of hardcoding an orientation and hoping, the final version
fits both orientations and only accepts whichever one produces
`user_factors.shape[0] == n_users` and `item_factors.shape[0] == n_items`.
If neither orientation is correct, it raises immediately with a clear error
rather than silently producing misaligned indices.

### 3. `ValueError: Buffer dtype mismatch, expected 'double' but got 'float'`

**Symptom:** `ItemItemRecommender.fit()` crashed inside `implicit`'s Cython
backend (`all_pairs_knn`).

**Root cause:** the interaction matrix was built with `dtype=np.float32`.
`implicit`'s `ItemItemRecommender` backend requires `float64` ("double")
specifically — `AlternatingLeastSquares` is more lenient about dtype, which
is why only the item-item model broke.

**Fix:** built `user_item_matrix` as `float64` from the start (Step 5 of
the notebook), so every downstream consumer — ALS, item-item fit, item-item
`.recommend()`, and the final production `recommend()` function — sees one
consistent, correct dtype. Added `assert user_item_matrix.dtype ==
np.float64` right after construction so this fails loudly at the source
instead of three steps later inside a library's Cython internals.

### 4. Choosing the right model for this dataset

Ran all four candidates through the same leakage-free evaluation
(Popularity, Transaction Popularity, User-User CF, Item-Item CF, ALS) with
identical Recall@5/10/20/50/100 metrics, rather than assuming an answer.

- **User-user CF performed worst.** ~80% of users have exactly one
  interaction, so cosine similarity between users is statistically
  unreliable — there's no real basis for "similar" at that sparsity.
- **ALS underperformed even the popularity baseline**, consistent with the
  same sparsity problem: too little signal per user for the latent factors
  to learn meaningful structure without further work (confidence weighting,
  much more tuning).
- **Item-item CF performed best**, because item vectors aggregate signal
  across many users even when individual users are near-cold-start —
  the item side of the matrix is simply denser than the user side here.

**Decision:** item-item collaborative filtering is the production model.
ALS and user-user CF are kept in the notebook as documented, honest
comparisons — not swept under the rug, but not shipped either.

## Architecture

```
RetailRocket data
      ↓
Chronological, leakage-free train/test split (per-user cutoff)
      ↓
Weighted interactions (view=1, addtocart=3, transaction=5)
      ↓
Item-item collaborative filtering (implicit.ItemItemRecommender, K=50)
      ↓
recommend(visitor_id, N) — same function used for evaluation and production
      ↓
Saved model artifacts (models/) — for FastAPI serving
```

## Model artifacts

Running the notebook's save step writes four files to `../models/`:

| File | Contents | Format |
|---|---|---|
| `item_model.pkl` | Fitted `ItemItemRecommender` (similarity matrix) | pickle |
| `user_item_matrix.npz` | Sparse interaction matrix, needed at inference time | scipy `.npz` |
| `mappings.pkl` | `user_to_idx`, `item_to_idx`, `idx_to_user`, `idx_to_item` | pickle |
| `popular_item_ids.pkl` | Popularity-ranked fallback for cold-start visitors | pickle |

To load and serve (e.g. from a FastAPI startup hook):

```python
import pickle
from scipy.sparse import load_npz

item_model = pickle.load(open('models/item_model.pkl', 'rb'))
user_item_matrix = load_npz('models/user_item_matrix.npz')
mappings = pickle.load(open('models/mappings.pkl', 'rb'))
popular_item_ids = pickle.load(open('models/popular_item_ids.pkl', 'rb'))
```

The notebook's Step 16 reloads these exact files and asserts the output
matches the in-memory model exactly before considering the save step
successful — a file existing on disk doesn't prove it's usable; matching
output does.

## Honest scope

**Built:** item-based collaborative filtering, evaluated against
popularity, transaction-popularity, user-user CF, and ALS baselines on a
leakage-free chronological split.

**Not built yet (planned next):** content-based NLP (BERT embeddings for
cold-start), sequential deep learning (SASRec, for session-aware
recommendations), RAG-based natural language product search, and GenAI
personalized copy. These are real next-phase work, not implemented here.

## Data

[RetailRocket e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
via Kaggle — `events.csv`, `item_properties.csv`, `category_tree.csv`.
