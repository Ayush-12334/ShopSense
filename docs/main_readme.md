# ShopSense — E-Commerce Recommendation System

ShopSense is a recommendation system built on the RetailRocket e-commerce dataset, combining **Item-Item Collaborative Filtering**, a sequence-aware transformer (**SASRec**), a popularity fallback, persistent user history, MLflow experiment tracking, and a Streamlit storefront.

The project is built around a question that matters more than any single model choice:

> **Does this system handle a brand-new user and a returning user differently, correctly, and provably?**

New users get cold-start popularity recommendations. As they interact, their history is persisted. Returning users get personalized recommendations from **Item-Item CF + SASRec**.

This is demonstrable end-to-end in the running application, not just claimed in a diagram.

> **Note on numbers:** dataset size is reported inconsistently across this project's history. Earlier notebook stages reported approximately 2.76M events, 1.1M users, and 210K items, while later processing produced different interaction/user counts. The current production catalog contains **234,561 items**. Earlier figures should therefore be treated as historical unless independently re-verified.

---

## What it does

```text
New visitor, no history
        │
        ▼
Popularity fallback
("popular with other shoppers")

Visitor views Product A
adds Product B to cart
views Product C
        │
        ▼
Interactions persisted to EventStore
(SQLite)

Same visitor returns later
with a new browser session
        │
        ▼
History retrieved from EventStore
        │
        ▼
Item-Item CF generates candidate products
        │
        ▼
SASRec reranks candidates using sequence context
        │
        ▼
Personalized recommendations
```

The core demonstration is:

```text
Cold Start
    ↓
User Behavior
    ↓
Persistent History
    ↓
New Session
    ↓
Personalized Recommendations
```

Every step is backed by a real data path through SQLite and the recommendation pipeline rather than relying on Streamlit session state.

---

# Architecture

```text
                         User
                           │
                           ▼
                 RecommendationService
                           │
                           ▼
                  RecommendationPipeline
                           │
              ┌────────────┴────────────┐
              │                         │
        New / Unknown User       Existing User
              │                         │
              ▼                         ▼
         Popularity              Item-Item CF
         Fallback                Candidate Generation
                                        │
                                        ▼
                                     SASRec
                                     Reranking
                                        │
                                        ▼
                              Final Recommendations
                                        │
                                        ▼
                                  Product Catalog
                                        │
                                        ▼
                                  Streamlit UI
```

### Three-layer architecture

```text
┌──────────────────────┐
│   Training Layer     │
│                      │
│ RetailRocket Data    │
│        ↓             │
│ Feature Engineering  │
│        ↓             │
│ Item-Item CF / SASRec│
│        ↓             │
│ MLflow               │
│        ↓             │
│ Model Artifacts      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Recommendation Layer │
│                      │
│ RecommendationService│
│        ↓             │
│ RecommendationPipeline│
│        ↓             │
│ Popularity / CF /    │
│ SASRec               │
│        ↓             │
│ Product Catalog      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Application Layer    │
│                      │
│ Streamlit UI         │
│        ↓             │
│ User Interaction     │
│        ↓             │
│ EventStore           │
│        ↓             │
│ Persistent History   │
└──────────────────────┘
```

Training is performed offline. The serving path does not retrain models when a user interacts with a product.

A user interaction results in a new event being stored in SQLite, while recommendations are generated using already-trained model artifacts.

---

# Problems I ran into, and how I solved them

Most of the engineering work in this project came from identifying and fixing issues that produced either incorrect results or misleading metrics.

## 1. Data leakage in the train/test split

### Symptom

A check comparing test targets against training history showed that approximately 98% of test items already appeared in the user's training history.

An early Item-Item CF evaluation also showed an unusual jump in recall as K increased.

### Cause

The initial split excluded only the exact held-out transaction.

Other events from the same user, including events occurring after the held-out transaction, remained in the training data.

That meant the model could indirectly train on future information.

### Fix

A per-user chronological cutoff was introduced.

For each user:

```text
User timeline

──────────── TRAIN ────────────│──── TEST ────
                               ↑
                            cutoff
```

Training interactions are restricted to events occurring strictly before the user's cutoff timestamp.

An assertion verifies this condition so that future events cannot silently enter training data.

---

## 2. ALS `KeyError` from inconsistent matrix orientation

### Symptom

The model produced:

```text
KeyError: 223332
```

when converting recommended indices back to item IDs.

### Cause

The matrix orientation and factor dimensions were not consistently aligned with the user/item mappings.

### Fix

Both possible orientations were checked and accepted only when the fitted factor dimensions matched the expected user/item counts.

The pipeline now validates the dimensions rather than silently accepting a mismatched mapping.

---

## 3. `Buffer dtype mismatch`

### Error

```text
ValueError: Buffer dtype mismatch,
expected 'double' but got 'float'
```

### Cause

The interaction matrix was created using `float32`.

The `implicit` Item-Item recommender's Cython implementation required `float64`.

### Fix

The interaction matrix is constructed using `float64`, with validation immediately after construction.

---

## 4. `filter_already_liked_items` padding problem

### Symptom

An early evaluation produced:

```text
Recall@5  = 0.0070
Recall@10 = 0.0090
Recall@20 = 0.0133
Recall@50 = 0.4987
Recall@100 = 0.6873
```

The large jump between Recall@20 and Recall@50 was suspicious.

### Cause

`implicit`'s:

```python
filter_already_liked_items=True
```

does not necessarily remove previously seen items from the returned array.

It can zero their scores and leave them as padding when real candidates run out.

Because many held-out targets were repeat interactions, this could artificially affect evaluation.

### Fix

Evaluation was changed to match the actual next-interaction task:

```text
exclude_seen=False
```

The resulting Item-Item CF recall curve became much smoother:

```text
Recall@5   ≈ 0.850
Recall@10  ≈ 0.891
Recall@20  ≈ 0.918
Recall@50  ≈ 0.942
Recall@100 ≈ 0.975
```

However, this led to an even more important finding.

---

# The trivial-baseline check

The most important evaluation finding came from testing whether the high recall was actually demonstrating collaborative-filtering intelligence.

A simple baseline was created that recommends items already present in the user's history, ordered by interaction weight.

```python
def repeat_own_history_recommend(user_idx, N):
    row = user_item_matrix[user_idx]
    order = np.argsort(-row.data)
    return list(row.indices[order][:N])
```

The result was approximately:

```text
Repeat-own-history

Recall@5  = 0.962
Recall@10 = 0.975
Recall@20 = 0.980
```

This exceeded Item-Item CF on the full repeat-heavy evaluation.

### What this means

The RetailRocket data contains substantial repeat behavior.

A model can therefore achieve high next-interaction recall simply by recommending products the user has already interacted with.

That means a high full-dataset recall number does not automatically prove that the recommendation model has learned useful collaborative relationships.

### Resulting evaluation change

The project therefore distinguishes between:

* **Repeat-interaction behavior**
* **Novel/discovery recommendations**

For repeat behavior, a simple history-based recommendation can be more appropriate than forcing a trained model to reproduce something a direct lookup already provides.

For discovery evaluation, the focus is on test cases where the target item is genuinely novel to the user's previous history.

This provides a more meaningful view of whether the recommendation model can discover products beyond the user's existing history.

---

# SASRec-specific design considerations

SASRec introduces additional sequence-modeling concerns.

## Causal masking

The transformer must not use future sequence positions when predicting the current item.

```text
Past interactions → allowed

Future interactions → blocked
```

This follows the same fundamental principle as the chronological data split: future information must not leak into prediction.

## Negative sampling

A sampled negative item must not actually be an item appearing in the user's sequence.

The negative sampler therefore excludes items already present in the sequence before accepting a negative sample.

## Graceful degradation

Not every user has enough session history for sequence-based reranking.

When SASRec cannot produce a valid sequence score:

```text
SASRec unavailable
       ↓
Fall back to Item-Item CF ordering
```

The recommendation system therefore degrades gracefully instead of failing.

---

# Model selection

Several approaches were evaluated using the same leakage-aware evaluation framework:

* Popularity
* User-User Collaborative Filtering
* Item-Item Collaborative Filtering
* ALS

The experiments showed that user-user collaborative filtering was affected by the sparsity of individual user histories, while Item-Item CF provided a stronger basis for candidate generation on this dataset.

ALS and User-User CF remain documented experiments rather than being part of the final recommendation path.

The deployed recommendation architecture uses:

```text
Popularity
    +
Item-Item CF
    +
SASRec
```

---

# Recommendation Models

## Popularity

Popularity provides the cold-start recommendation strategy.

It is used for:

* New users
* Unknown users
* Users with insufficient history
* Personalized recommendation fallback cases

The popularity model incorporates interaction strength and recency.

```text
User has insufficient history
            ↓
       Popularity Model
            ↓
      Popular Products
```

---

## Item-Item Collaborative Filtering

Item-Item CF is the primary candidate-generation model.

It learns relationships between products based on user interaction patterns.

Interaction weights:

| Event       | Weight |
| ----------- | -----: |
| View        |      1 |
| Add to Cart |      3 |
| Transaction |      5 |

The basic recommendation flow is:

```text
User History
     ↓
Previously interacted items
     ↓
Item-Item similarities
     ↓
Candidate products
```

The current pipeline generates approximately 200 candidates before SASRec reranking.

---

## SASRec

SASRec is a sequence-aware transformer model.

Unlike plain item similarity, SASRec considers the order of interactions.

For example:

```text
Product A
    ↓
Product B
    ↓
Product C
    ↓
Product D
```

The sequence provides information about the user's recent behavior.

In ShopSense, SASRec is primarily used as a **reranker** rather than a standalone candidate generator.

```text
Item-Item CF
     ↓
~200 candidates
     ↓
SASRec
     ↓
Reranked candidates
     ↓
Top-N
```

If a usable sequence is unavailable, the pipeline can fall back to Item-Item CF ordering.

---

# Persistent History and Cross-Session Personalization

User interactions are stored in SQLite through the EventStore.

Supported interaction types include:

* `view`
* `add_to_cart`
* `transaction`

```text
Session 1
User 100
   │
   ├── view
   ├── add_to_cart
   └── transaction
           │
           ▼
     EventStore
           │
           ▼
shopsense_events.db


New Session
User 100
   │
   ▼
Previous history retrieved
   │
   ▼
RecommendationPipeline
   │
   ├── Item-Item CF
   └── SASRec
   │
   ▼
Personalized Recommendations
```

The history is not dependent on Streamlit's temporary `session_state`.

Starting a new application session therefore does not erase the user's persisted interaction history.

---

# RecommendationService

`RecommendationService` provides the application-facing interface for recommendations.

The Streamlit application does not directly operate the recommendation models.

```text
Streamlit
    ↓
RecommendationService
    ↓
RecommendationPipeline
    ↓
Models
```

The service is responsible for:

* Requesting recommendations
* Logging user interactions
* Working with persistent history
* Connecting application behavior to the recommendation pipeline

This keeps the UI and recommendation logic separated.

---

# SessionManager

`SessionManager` handles application session information.

A new session can be created while retaining the same user ID.

```text
User 100

Session 1
   ↓
Interactions
   ↓
EventStore

New Session
   ↓
User 100
   ↓
Stored history retrieved
   ↓
Personalized recommendations
```

---

# Product Catalog

RetailRocket provides anonymized item IDs and hashed item properties rather than human-readable product descriptions.

`CatalogBuilder` combines the available item metadata with the trained item vocabulary to create a usable product catalog.

The resulting catalog is stored as:

```text
product_catalog.parquet
```

The catalog can contain information such as:

* RetailRocket item ID
* Model item index
* Category information
* Availability
* Display name
* Synthetic-name indicator

Where reliable product naming information is unavailable, the system generates a synthetic display name and explicitly marks it as synthetic rather than presenting it as original RetailRocket metadata.

---

# MLflow

ShopSense uses MLflow for experiment tracking and model management.

The recommendation system tracks models including:

* Item-Item CF
* SASRec
* Popularity

The general workflow is:

```text
Training / Experiments
        ↓
      MLflow
        │
        ├── Item-Item CF
        ├── SASRec
        └── Popularity
        │
        ▼
     Artifacts
        │
        ▼
Recommendation Pipeline
```

The local MLflow backend uses:

```text
sqlite:///mlflow.db
```

The project experiment is:

```text
shopsense-production
```

---

# Streamlit Storefront

The Streamlit application provides the user-facing recommendation interface.

The main flow is:

```text
Enter User ID
      ↓
Show Recommendations
      ↓
Display Recommendation Model
      ↓
Display Products
      ↓
View / Buy
      ↓
Record Interaction
      ↓
Refresh Recommendations
```

The recommendation source is returned by the backend rather than being hard-coded by the UI.

For example:

```text
New / insufficient-history user
        ↓
🔥 Popularity


Existing user with usable history
        ↓
🧠 Item-Item CF + SASRec
```

Run the application with:

```bash
streamlit run app.py
```

---

# Evaluation

The main Item-Item CF evaluation results are:

| Metric     | Item-Item CF |
| ---------- | -----------: |
| Recall@5   |        0.850 |
| Recall@10  |        0.891 |
| Recall@20  |        0.918 |
| Recall@50  |        0.942 |
| Recall@100 |        0.975 |

These numbers are **dataset- and split-specific**.

They should not be interpreted as general real-world recommendation performance.

Because the full evaluation is heavily affected by repeat interactions, the project also tracks discovery-oriented evaluation separately.

Additional metrics include:

* Coverage@K
* Diversity@K
* Novelty@K

The evaluation framework is designed to distinguish between simply repeating known products and discovering genuinely new products.

---

# Testing

The project includes automated tests covering the recommendation system.

The test suite checks areas including:

* Prediction pipeline loading
* Anonymous users
* New users
* Low-history users
* Known-user personalization
* Stored user history
* Cross-session personalization
* Recommendation generation

Run the tests with:

```bash
python -m tests.test_recommendation_system
```

The intended test flow is:

```text
Prediction Pipeline
        ↓
Recommendation Models
        ↓
User History
        ↓
RecommendationService
        ↓
Recommendations
```

---

# Project Structure

The repository is organized around the training pipeline, prediction pipeline, database layer, recommendation service, and application.

```text
shope_sense/
│
├── app.py
│   └── Streamlit storefront
│
├── datasets/
│   └── RetailRocket dataset
│       ├── events.csv
│       ├── item_properties_part1.csv
│       ├── item_properties_part2.csv
│       └── category_tree.csv
│
├── notebooks/
│   └── Exploration, preprocessing,
│       model experiments and evaluation
│
├── src/
│   │
│   ├── components/
│   │   ├── DataIngestion
│   │   ├── FeatureEngineering
│   │   ├── ModelTrainer
│   │   └── CatalogBuilder
│   │
│   ├── configuration/
│   │   └── Configuration and artifact settings
│   │
│   ├── database/
│   │   ├── EventStore
│   │   └── services/
│   │       ├── recommendation_service.py
│   │       └── session_manager.py
│   │
│   ├── pipeline/
│   │   ├── TrainingPipeline
│   │   ├── PredictionPipeline
│   │   └── RecommendationPipeline
│   │
│   ├── model/
│   │   └── Model implementations / wrappers
│   │
│   ├── utils/
│   │   └── Utility functions
│   │
│   ├── logger/
│   │   └── Logging configuration
│   │
│   └── exception/
│       └── Custom exception handling
│
├── models/
│   └── Local model artifacts
│
├── mlartifacts/
│   └── MLflow artifacts
│
├── tests/
│   └── Recommendation-system tests
│
├── docs/
│   └── Project documentation
│
├── scope/
│   └── Project scope and planning
│
├── logs/
│   └── Runtime logs
│
├── item_to_idx.pkl
├── last_session_by_user.pkl
├── product_catalog.parquet
├── shopsense_events.db
├── test_shopsense_events.db
├── mlflow.db
├── .gitignore
└── README.md
```

> **Important:** generated runtime files such as SQLite databases, MLflow databases, logs, virtual environments, and large model artifacts should only be committed if they are intentionally part of the repository. `.gitignore` should control these files appropriately.

---

# Technology Stack

### Machine Learning

* Python
* scikit-learn
* `implicit`
* PyTorch

### Recommendation Systems

* Item-Item Collaborative Filtering
* SASRec
* Popularity-based recommendation

### Data Processing

* pandas
* NumPy
* Parquet

### Application

* Streamlit

### Storage

* SQLite
* Parquet
* MLflow artifact storage

### Experiment Tracking

* MLflow

### Development

* Git
* Python virtual environment
* Automated testing

---

# Local Setup

Clone the repository:

```bash
git clone <your-repository-url>
cd shope_sense
```

Create the virtual environment:

```bash
python -m venv shope_sense
```

Activate it on Windows PowerShell:

```powershell
.\shope_sense\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the recommendation tests:

```bash
python -m tests.test_recommendation_system
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

# Honest Scope

## Built

* Item-Item Collaborative Filtering
* SASRec sequence-aware reranking
* Popularity fallback
* Transaction-weighted interactions
* Leakage-free chronological evaluation
* Repeat-vs-novel evaluation analysis
* Trivial repeat-history baseline
* Recall@K evaluation
* Coverage@K
* Diversity@K
* Novelty@K
* Product catalog construction
* Recommendation / Prediction Pipeline
* Persistent EventStore
* Cross-session personalization
* MLflow experiment tracking
* MLflow model registration
* RecommendationService
* SessionManager
* Streamlit storefront
* Automated recommendation-system tests

## Planned / Not Yet Built

The following are future extensions and are **not currently load-bearing parts of ShopSense**:

* Content-based item embeddings
* BERT-based product representations
* Vector search
* ChromaDB
* RAG-based natural-language product search
* GenAI-generated recommendation explanations
* Cloud deployment
* Production monitoring
* Observability
* A/B testing

The architecture is intentionally layered so these capabilities can be added later without replacing the existing recommendation system.

---

# Future Architecture

```text
                         ShopSense
                            │
             ┌──────────────┴──────────────┐
             │                             │
      Recommendation                 Product Search
          System                         (planned)
             │                             │
      ┌──────┼──────┐                     ▼
      │      │      │                Embeddings
 Popularity  CF   SASRec                 │
                                         ▼
                                     Vector DB
                                         │
                                         ▼
                                        RAG
                                         │
                                         ▼
                              Natural Language UI
```

The current recommendation system remains independent of the future GenAI/search layer.

---

# Data

The project uses the RetailRocket e-commerce dataset.

The repository expects:

```text
events.csv
item_properties_part1.csv
item_properties_part2.csv
category_tree.csv
```

Dataset source:

[RetailRocket E-Commerce Dataset — Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

The dataset contains anonymized e-commerce interaction data including views, add-to-cart events, and transactions.

---

# Further Documentation

Project-specific documentation can be maintained under the `docs/` directory.

Recommended documentation files include:

```text
docs/
├── Evaluation_Improvements_Log.md
└── SASRec_Implementation_Notes.md
```

These can contain the detailed evaluation history, implementation decisions, and debugging notes without making the main README unnecessarily difficult to navigate.

---

# Project Goal

ShopSense is designed to demonstrate the complete path from recommendation-system experimentation to an application-oriented machine-learning system.

```text
RetailRocket Data
        ↓
Data Processing
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Evaluation
        ↓
MLflow
        ↓
Prediction Pipeline
        ↓
Persistent User History
        ↓
RecommendationService
        ↓
Streamlit Storefront
```

The central demonstration is:

```text
New User
   ↓
Popularity

        ↓ user interactions

Persistent History
   ↓
New Session
   ↓
Item-Item CF
   ↓
SASRec
   ↓
Personalized Recommendations
```

This makes the project more than a standalone recommendation model: it demonstrates how trained models, persistent state, inference logic, evaluation, experiment tracking, and a user-facing application fit together.
