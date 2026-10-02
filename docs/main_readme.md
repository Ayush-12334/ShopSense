# ShopSense — Collaborative Filtering & Sequential Recommendation Engine

ShopSense is a recommendation system built on the **RetailRocket e-commerce dataset** (~2.76M events, ~1.1M users, ~235K items).

The project started as an **item-based collaborative filtering recommender** and evolved into a complete recommendation application with:

* Item-Item Collaborative Filtering for candidate generation
* SASRec for sequential reranking
* Popularity fallback for cold-start users
* Persistent cross-session user history
* Event-based interaction storage
* Product catalog mapping and metadata
* MLflow model tracking and registration
* Recall, coverage, and diversity evaluation
* Streamlit application for interacting with the recommender

The main goal was not simply to train a recommendation model, but to understand the engineering problems involved in taking a recommender from an offline notebook to a reloadable, stateful application.

---

## What it does

ShopSense changes its recommendation strategy depending on the amount of user history available.

```text
New / Unknown User
        ↓
Popularity-based recommendations

Returning User with insufficient history
        ↓
Popularity fallback

Returning User with usable history
        ↓
Item-Item Collaborative Filtering
        ↓
Candidate pool (~200 items)
        ↓
SASRec sequential reranking
        ↓
Top-N personalized recommendations
```

For example:

```text
User: 172
Session: new session

        ↓

Stored interaction history retrieved
        ↓
Item-Item CF generates candidate products
        ↓
SASRec considers the user's recent sequence
        ↓
Final personalized recommendations
```

The same user can start a new session and still receive personalized recommendations because their interaction history is stored outside the Streamlit session.

---

# Problems I ran into, and how I solved them

This project went through several real debugging and evaluation rounds. They are documented here because diagnosing and fixing these problems was a significant part of the engineering work.

---

## 1. Data leakage in the train/test split

### Symptom

A check comparing test targets against training history showed approximately 98% of test items already appeared in the user's "training" data.

Item-Item Recall@50 also showed an unusual jump from approximately 1% to approximately 39%.

### Root cause

The original train/test split excluded only the exact:

```text
(visitorid, itemid, timestamp)
```

row corresponding to the held-out transaction.

Other events from the same user, including events that happened **after** the held-out transaction, remained in the training data.

The model was therefore being trained on future information.

### Fix

The split was changed to a **per-user chronological cutoff**.

For every evaluation user:

```text
cutoff = timestamp of the held-out target

training events
    < cutoff

test target
    = cutoff
```

An assertion was also added to verify that training events do not cross the cutoff.

This made the evaluation leakage-free with respect to future events.

One important distinction is that browsing before a purchase is not leakage.

For example:

```text
View item 10034
      ↓
Add to cart
      ↓
Transaction
```

If the view and add-to-cart occurred before the evaluation cutoff, they are valid historical signals for predicting the later interaction.

---

# 2. ALS KeyError caused by matrix orientation

### Symptom

The ALS implementation produced errors such as:

```text
KeyError: 223332
```

when converting recommendation indices back to item IDs.

### Root cause

The interaction matrix orientation and the factor dimensions were not consistently aligned with the mappings.

Different `implicit` model APIs and versions can make matrix orientation easy to misuse.

### Fix

The final implementation validates the fitted factor dimensions against the expected number of users and items.

The model is only accepted when:

```text
user_factors.shape[0] == n_users

item_factors.shape[0] == n_items
```

If the dimensions do not match, the pipeline fails explicitly instead of silently producing incorrect mappings.

ALS was ultimately kept as an evaluation comparison rather than the production recommender.

---

# 3. `Buffer dtype mismatch` in Item-Item CF

### Symptom

`ItemItemRecommender.fit()` failed inside the `implicit` Cython backend with:

```text
ValueError:
Buffer dtype mismatch, expected 'double' but got 'float'
```

### Root cause

The interaction matrix was created using:

```python
dtype=np.float32
```

while the Item-Item recommender backend expected `float64`.

### Fix

The interaction matrix was changed to:

```python
dtype=np.float64
```

An explicit assertion was also added:

```python
assert user_item_matrix.dtype == np.float64
```

This ensures the problem is detected at matrix construction rather than later inside a library implementation.

---

# 4. Choosing the recommendation model

Several recommendation approaches were evaluated using the same chronological evaluation framework.

The evaluated approaches included:

* Popularity
* Transaction Popularity
* User-User Collaborative Filtering
* Item-Item Collaborative Filtering
* ALS

### Observation

Approximately 80% of users have only one interaction in the RetailRocket data.

This makes user-user similarity difficult to learn because many users do not have enough interaction history to establish meaningful similarity.

ALS also did not outperform the simpler popularity baseline under the initial configuration.

Item-Item CF provided stronger collaborative signal because item relationships aggregate interactions across many users.

### Decision

Item-Item CF became the collaborative filtering component of the production recommendation pipeline.

However, instead of using Item-Item CF directly as the final ranking model, it is now used to generate a candidate pool.

```text
User history
     ↓
Item-Item CF
     ↓
~200 candidates
     ↓
SASRec
     ↓
Final ranking
```

This separates **candidate generation** from **sequential ranking**.

---

# 5. Sequential recommendation with SASRec

After implementing Item-Item CF, the next problem was that collaborative filtering alone does not explicitly model the order of a user's recent interactions.

For example:

```text
User history:

Item A → Item B → Item C → Item D
```

The most recent sequence can contain information about the user's current intent.

### Implementation

SASRec was implemented as the sequential reranking component.

The model uses:

* Transformer encoder architecture
* Causal attention masking
* Maximum sequence length of 50
* Random negative sampling
* Negative samples excluded from the user's observed sequence
* Binary classification objective
* Learning rate of `1e-3`
* Batch size of `128`
* GPU when available, otherwise CPU

### Role of SASRec

SASRec does not search the entire product catalog.

Instead:

```text
Item-Item CF
    ↓
Candidate pool
    ↓
SASRec reranking
    ↓
Top-N
```

This keeps the sequential model focused on ranking a manageable candidate set.

---

# 6. Cross-session personalization

A recommendation system should not forget a user simply because they started a new application session.

An early implementation relied too heavily on application/session state.

The final architecture separates:

```text
Application session
```

from:

```text
Persistent user interaction history
```

### Final flow

```text
User interacts with products
        ↓
EventStore
        ↓
Persistent interaction history
        ↓
New application session
        ↓
History retrieved using User ID
        ↓
Recommendation pipeline
```

The interaction history is stored in SQLite through the `EventStore`.

This means:

```text
Session 1
User 123
   ↓
View A
View B
Add to cart C

        ↓ new session

Session 2
User 123
   ↓
Previous history retrieved
   ↓
Personalized recommendations
```

The Streamlit session is therefore not treated as the source of truth for user history.

---

# 7. Cold-start and insufficient-history handling

SASRec and collaborative filtering require historical interactions.

A completely new user does not provide enough information for personalized sequence modeling.

ShopSense therefore implements graceful fallback behavior.

### New / unknown user

```text
No history
   ↓
Popularity model
```

### Returning user with insufficient history

```text
Very short history
   ↓
Popularity fallback
```

### Returning user with usable history

```text
History
   ↓
Item-Item CF
   ↓
SASRec
   ↓
Personalized recommendations
```

The recommendation pipeline therefore does not fail simply because a user has no history.

---

# 8. Persistent event storage

User interactions are stored using an SQLite-backed `EventStore`.

The system records interaction events such as:

```text
view
click
add_to_cart
transaction
```

The events are weighted during recommendation processing.

Current interaction weights are:

```text
view         = 1
click        = 2
add_to_cart  = 3
transaction  = 5
```

This allows stronger actions such as purchases and cart additions to carry more signal than simple views.

The EventStore is also used by the application to record new interactions generated through the Streamlit interface.

---

# 9. Product catalog integration

The recommendation models operate using internal model item indices.

Those indices are not useful to a user-facing application by themselves.

A product catalog layer was therefore added to map:

```text
model item index
       ↓
RetailRocket item ID
       ↓
product metadata
```

The catalog contains available product metadata and category information where available.

For products without complete metadata, the application can fall back to generated display information rather than failing to display the recommendation.

This allows the recommendation model and the user-facing application to remain separated.

---

# 10. Evaluation beyond Recall

Recall alone does not completely describe the quality of a recommendation system.

A system could achieve high recall while repeatedly recommending a very small portion of the catalog.

ShopSense therefore evaluates additional properties of the recommendation output.

## Recall

Recall@K measures whether the target item appears in the top-K recommendations.

The project evaluates:

```text
Recall@5
Recall@10
Recall@20
Recall@50
Recall@100
```

Both repeat-heavy recommendation behavior and discovery-oriented evaluation are considered separately.

---

## Coverage

Coverage measures how much of the available catalog is actually being recommended.

Conceptually:

```text
Coverage =
unique recommended items
------------------------
available catalog items
```

This helps identify whether the recommender is concentrated on a small group of popular products.

For example, a system recommending the same 100 products to nearly everyone may have useful recall for popular targets but poor catalog coverage.

Coverage is therefore tracked alongside recall rather than relying on recall alone.

---

## Diversity

Diversity measures how different the recommended products are from one another.

A recommendation list such as:

```text
Product A
Product A variant
Product A variant
Product A variant
Product A variant
```

may have useful relevance but provides little variety.

The project therefore evaluates recommendation-list diversity to identify whether the final ranking is overly concentrated.

Coverage and diversity are treated as complementary diagnostics:

```text
Recall
  +
Coverage
  +
Diversity
```

This provides a more complete view of recommendation behavior than a single accuracy metric.

---

# 11. Repeat-history baseline

Another important finding was that the RetailRocket dataset is heavily repeat-oriented.

A simple baseline that recommends items already seen in a user's history achieved very high Recall@K on the repeat-heavy evaluation.

For example, the repeat-history baseline achieved approximately:

```text
Recall@5  ≈ 0.962
```

This revealed an important evaluation issue:

> High recall does not necessarily mean the recommender is discovering new products.

Therefore the project distinguishes between:

### Repeat behavior

```text
"Recommend things the user already interacted with."
```

and:

### Discovery behavior

```text
"Recommend relevant items that the user has not already interacted with."
```

Both are useful behaviors, but they answer different evaluation questions.

---

# 12. MLflow model management

The project uses MLflow to track model runs and artifacts.

The recommendation models are associated with training runs rather than being treated as unmanaged local files.

The current MLflow setup tracks models including:

```text
Item-Item CF
SASRec
Popularity
```

The prediction pipeline loads the required artifacts from the configured/latest training run.

This makes the transition from:

```text
training
   ↓
artifact
   ↓
prediction
```

explicit rather than hard-coding a particular local model file.

---

# 13. Prediction Pipeline

The final prediction layer provides a common entry point for generating recommendations.

Conceptually:

```text
PredictionPipeline
       ↓
Load configuration
       ↓
Load catalog
       ↓
Load recommendation artifacts
       ↓
Load persistent user history
       ↓
Generate recommendations
```

The pipeline handles:

* Unknown users
* Known users
* Insufficient history
* Item-Item CF candidate generation
* SASRec reranking
* Popularity fallback
* Product ID mapping
* Recommendation metadata

The recommendation source is returned with the output so the application does not need to guess which model produced the result.

---

# 14. Streamlit application

A Streamlit interface was added to make the recommendation system interactively testable.

The application allows a user to:

```text
Enter User ID
      ↓
Get recommendations
      ↓
View recommendation source
      ↓
Interact with products
      ↓
Record events
      ↓
Generate updated recommendations
```

The UI can show whether recommendations came from:

```text
🔥 Popularity

🧠 Item-Item CF + SASRec

🧠 Session Personalization
```

The application therefore acts as a practical demonstration of the complete recommendation pipeline rather than only displaying offline evaluation metrics.

---

# Final Recommendation Architecture

The current architecture is:

```text
                    RetailRocket Dataset
                            │
                            ▼
              Chronological Data Preparation
                            │
                            ▼
                  Weighted Interactions
                            │
                            ▼
                    Model Training
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        Item-Item CF      SASRec      Popularity
              │             │             │
              └──────┬──────┘             │
                     │                    │
                     ▼                    │
             Candidate Generation        │
                     │                    │
                     ▼                    │
                SASRec Ranker             │
                     │                    │
                     └─────────┬──────────┘
                               ▼
                       Prediction Pipeline
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
              Persistent History       Catalog
                    │                     │
                    └──────────┬──────────┘
                               ▼
                       Recommendation API
                               │
                               ▼
                         Streamlit App
```

For a new user:

```text
User
 ↓
No history
 ↓
Popularity
```

For a returning user:

```text
User
 ↓
EventStore
 ↓
Historical interactions
 ↓
Item-Item CF
 ↓
Candidate pool
 ↓
SASRec
 ↓
Top-N recommendations
```

---

# What is currently implemented

### Recommendation

* [x] Item-Item Collaborative Filtering
* [x] Popularity fallback
* [x] Weighted interaction signals
* [x] SASRec sequential reranking
* [x] Candidate generation + reranking architecture
* [x] Cold-start handling
* [x] Insufficient-history fallback

### Data & Evaluation

* [x] Chronological train/test split
* [x] Leakage checks
* [x] Recall@5/10/20/50/100
* [x] Repeat-history baseline
* [x] Discovery-oriented evaluation
* [x] Recommendation coverage
* [x] Recommendation diversity
* [x] Model comparison

### Application

* [x] Persistent EventStore
* [x] Cross-session personalization
* [x] Product catalog integration
* [x] Prediction Pipeline
* [x] Recommendation Service
* [x] Streamlit application
* [x] Interaction logging

### MLOps / Engineering

* [x] MLflow experiment tracking
* [x] Registered recommendation models
* [x] Reloadable model artifacts
* [x] Configuration-driven prediction pipeline
* [x] Automated recommendation-system tests

---

# Honest Scope

The current project is a working recommendation-system application built around:

```text
Item-Item CF
      +
SASRec
      +
Popularity fallback
      +
Persistent user history
      +
Product catalog
      +
Evaluation
      +
Streamlit
      +
MLflow
```

The project intentionally focuses on recommendation-system engineering and does not claim that every possible production feature has been implemented.

Potential future work includes:

* Content-based recommendation using product embeddings
* Better cold-start recommendations using product metadata
* RAG-based natural-language product search
* Natural-language recommendation explanations
* More advanced ranking objectives
* Online model monitoring
* Cloud deployment
* Automated retraining
* A production API layer

These are future extensions rather than components currently claimed as implemented.

---

# Data

The project uses the **RetailRocket e-commerce dataset**, containing:

* `events.csv`
* `item_properties.csv`
* `category_tree.csv`

The dataset contains approximately:

```text
2.76M events
~1.4M visitors
~235K items
```

The recommendation experiments use chronological interaction history rather than randomly shuffling events, so that evaluation better represents the temporal nature of recommendation.

---

# Technology Stack

```text
Python
Pandas
NumPy
SciPy
Scikit-learn
Implicit
PyTorch
MLflow
SQLite
Streamlit
PyArrow
YAML
```

The project is structured so that the recommendation models, persistence layer, prediction pipeline, and application layer remain separate.

---

# Project Goal

The main goal of ShopSense is to understand the complete path from:

```text
Raw interaction data
        ↓
Data preparation
        ↓
Leakage-free evaluation
        ↓
Recommendation models
        ↓
Model selection
        ↓
Candidate generation
        ↓
Sequential reranking
        ↓
Persistent user history
        ↓
Prediction pipeline
        ↓
Evaluation
        ↓
User-facing application
```

The project therefore focuses not only on obtaining a recommendation metric, but on understanding **why a recommendation system behaves the way it does and how it can be turned into a usable application**.
