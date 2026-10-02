# ShopSense — E-Commerce Recommendation System

An end-to-end e-commerce recommendation system built on the **RetailRocket dataset**, combining **Item-Item Collaborative Filtering, SASRec sequential reranking, popularity-based fallback recommendations, persistent user history, MLflow model management, and a Streamlit storefront**.

ShopSense is designed not only as a recommendation model, but as a complete machine-learning application that covers the path from raw interaction data to model training, evaluation, prediction, persistence, and user-facing recommendations.

---

## What Problem Does ShopSense Solve?

An e-commerce recommendation system cannot treat every visitor the same way.

Consider three users:

* A **new visitor** with no interaction history
* A **returning user** who has interacted with only a few products
* A **returning user** with enough historical activity to personalize recommendations

ShopSense handles these cases differently:

```text
                         ┌─────────────────────┐
                         │      User ID        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Persistent History  │
                         │    in SQLite        │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
              No / little history             Usable history
                    │                               │
                    ▼                               ▼
              Popularity                 Item-Item CF Candidates
              Fallback                           │
                                                 ▼
                                          SASRec Reranking
                                                 │
                                                 ▼
                                      Personalized Top-N
```

The system therefore provides a graceful recommendation strategy rather than forcing one model to handle every user.

---

# Recommendation Flow

## 1. New or Unknown User

A user without sufficient historical interactions cannot be reliably personalized.

ShopSense uses a popularity-based recommender:

```text
New User
   ↓
No usable history
   ↓
Popularity Model
   ↓
Top-N Products
```

This provides a practical cold-start fallback.

---

## 2. Returning User With Insufficient History

A returning user may exist in the system but still have too little history for sequential modeling.

Instead of producing unstable personalized recommendations, ShopSense falls back to popularity:

```text
Returning User
      ↓
Insufficient History
      ↓
Popularity Fallback
      ↓
Top-N Products
```

---

## 3. Returning User With Usable History

For users with enough interaction history:

```text
User History
     ↓
Item-Item Collaborative Filtering
     ↓
Candidate Pool
     ↓
SASRec Sequential Reranking
     ↓
Top-N Recommendations
```

Item-Item CF generates a relatively small candidate set, while SASRec uses the user's recent sequence to determine which candidates are more relevant.

This avoids requiring SASRec to score the entire product catalog.

---

# Key Features

* Item-Item Collaborative Filtering
* SASRec sequential recommendation model
* Popularity-based cold-start fallback
* Persistent user interaction history
* Cross-session personalization
* Product catalog integration
* MLflow experiment tracking and model registration
* Streamlit e-commerce storefront
* Leakage-free chronological evaluation
* Recall@K evaluation
* Coverage, diversity, and novelty evaluation
* Automated recommendation-system tests
* Graceful fallback when personalization is unavailable

---

# Architecture

```text
                         ┌──────────────────────┐
                         │   RetailRocket Data  │
                         └───────────┬──────────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │   Data Ingestion     │
                         └───────────┬──────────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │ Feature Engineering │
                         └───────────┬──────────┘
                                     │
                  ┌──────────────────┼──────────────────┐
                  │                  │                  │
                  ▼                  ▼                  ▼
          ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
          │ Popularity   │   │ Item-Item CF │   │   SASRec     │
          │    Model     │   │    Model     │   │    Model     │
          └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
                 │                  │                  │
                 │                  └────────┬─────────┘
                 │                           │
                 │                           ▼
                 │                  ┌──────────────────┐
                 │                  │ Prediction /     │
                 │                  │ Recommendation   │
                 │                  │ Pipeline         │
                 │                  └────────┬─────────┘
                 │                           │
                 └──────────────┬────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Recommendation   │
                       │    Service       │
                       └────────┬─────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
             ┌──────────────┐       ┌──────────────┐
             │ Event Store  │       │  Streamlit   │
             │   SQLite     │       │  Storefront  │
             └──────────────┘       └──────────────┘
```

---

# Model Strategy

## Popularity Model

Popularity is used for:

* New users
* Unknown users
* Users with insufficient history
* Cases where personalized recommendation generation cannot be performed

The popularity model provides a reliable fallback instead of returning an empty recommendation list.

---

## Item-Item Collaborative Filtering

Item-Item CF is the primary candidate-generation model.

The basic idea is:

```text
User interacted with:
    Item A
    Item B
    Item C

Find items similar to:
    A + B + C

Generate candidate pool
        ↓
Pass candidates to SASRec
```

The model works on user-item interaction data and generates a candidate pool before sequential reranking.

The production recommendation pipeline uses approximately **200 candidates** before SASRec reranking.

---

## SASRec

SASRec is used as a **sequential reranking model**.

Rather than generating recommendations across the entire catalog, it receives candidate products from Item-Item CF and scores them according to the user's recent interaction sequence.

### Configuration

* Maximum sequence length: **50**
* Batch size: **128**
* Learning rate: **1e-3**
* Objective: **binary classification / BCE-style training**
* Negative sampling: random negatives excluding observed sequence items
* Causal attention masking
* GPU used when available
* CPU fallback when GPU is unavailable

Conceptually:

```text
User sequence

[Item 12, Item 45, Item 81, Item 103, ...]

                    ↓

                 SASRec

                    ↓

Candidate scores

Item 201 → 0.82
Item 304 → 0.71
Item 156 → 0.65
...
```

The highest-scoring candidates are returned to the application.

---

# Why Item-Item CF + SASRec?

The models serve different purposes.

| Component             | Responsibility                                        |
| --------------------- | ----------------------------------------------------- |
| Popularity            | Cold-start and fallback recommendations               |
| Item-Item CF          | Generate relevant candidate products                  |
| SASRec                | Understand recent user sequence and rerank candidates |
| EventStore            | Persist interaction history                           |
| RecommendationService | Coordinate recommendation logic                       |
| Streamlit             | User-facing application                               |

This separation keeps candidate generation, ranking, persistence, and presentation independent.

---

# Persistent User History

One of the important features of ShopSense is that user history is **not dependent on Streamlit's in-memory session state**.

Interaction history is stored in a SQLite EventStore.

```text
User
 │
 ├── Session 1
 │      ├── view Item A
 │      ├── view Item B
 │      └── transaction Item C
 │
 └── Session 2
        │
        └── Previous history available
                ↓
        Personalized recommendations
```

This allows a user to start a new application/browser session while retaining their previous interaction history.

---

# Cross-Session Personalization

ShopSense explicitly tests whether historical interactions are reused across sessions.

Example:

```text
Session 1
─────────

User: 123

View:
    Product A
    Product B
    Product C

        ↓

Events stored in SQLite
```

Then:

```text
Session 2
─────────

User: 123

New session
    ↓
Load persistent history
    ↓
Use previous interactions
    ↓
Generate personalized recommendations
```

The recommendation system therefore separates:

* **application session state**
* **persistent user interaction history**

This is important for a realistic e-commerce recommendation workflow.

---

# Interaction Tracking

The application records user interactions in the EventStore.

Current application-level interactions include:

| Interaction | Weight |
| ----------- | -----: |
| View        |      1 |
| Add to Cart |      3 |
| Transaction |      5 |

Higher-value interactions receive greater importance when constructing the interaction representation.

The transaction signal is therefore stronger than a simple product view.

---

# Product Catalog

Recommendation models operate using item identifiers, but a storefront needs product information that users can understand.

ShopSense includes a `CatalogBuilder` that connects model item IDs with product metadata.

The catalog contains:

* Product/item ID
* Category information
* Availability information where present
* Display name
* Synthetic metadata indicator

The generated catalog currently contains approximately **234,561 product rows**.

Some RetailRocket products do not have complete metadata. For those products, ShopSense generates placeholder display names.

These generated names are explicitly marked as:

```text
is_synthetic = True
```

They are therefore not presented as original RetailRocket product names.

---

# Evaluation

Recommendation systems should not be evaluated using accuracy alone.

ShopSense evaluates the recommendation pipeline using:

* Recall@K
* Coverage@K
* Diversity@K
* Novelty@K

---

## Leakage-Free Evaluation

The original evaluation process exposed an important problem.

A naive split can accidentally allow future interactions from a user to remain inside the training data.

For a chronological recommendation task, this creates temporal leakage.

The corrected evaluation uses a **per-user chronological cutoff**:

```text
User timeline

t1 ── t2 ── t3 ── t4 ── t5

Training:
t1 ── t2 ── t3

Evaluation:
          t4 ── t5
```

The pipeline verifies that training events occur before the evaluation cutoff.

This makes the evaluation much closer to the actual recommendation scenario.

---

# Evaluation Results

The Item-Item CF model achieved approximately:

| Metric     | Recall |
| ---------- | -----: |
| Recall@5   |  0.850 |
| Recall@10  |  0.891 |
| Recall@20  |  0.918 |
| Recall@50  |  0.942 |
| Recall@100 |  0.975 |

These results should be interpreted in the context of the RetailRocket interaction distribution and the evaluation protocol.

---

# The Repeat-Interaction Baseline

During evaluation, a simple baseline produced surprisingly high recall.

The baseline recommends items from a user's previous interaction history.

Approximate results:

| Metric    | Repeat-History Recall |
| --------- | --------------------: |
| Recall@5  |                 0.962 |
| Recall@10 |                 0.975 |
| Recall@20 |                 0.980 |

This revealed an important characteristic of the dataset:

> Many evaluation targets are repeat interactions.

Therefore, a high Recall@K value does not automatically mean that a recommendation model is discovering new products.

ShopSense consequently distinguishes between:

### Repeat recommendation

```text
User previously interacted with Product A
                 ↓
Recommend Product A again
```

and:

### Discovery recommendation

```text
User previously interacted with Product A
                 ↓
Recommend a new, relevant Product B
```

This distinction is important when interpreting recommendation-system performance.

---

# Discovery Evaluation

ShopSense also supports evaluation where previously seen items are excluded from recommendation results.

This provides a separate view of the model's ability to recommend products that are new to the user.

The project therefore avoids relying on a single metric to describe recommendation quality.

---

# Other Models Investigated

Several approaches were explored during development.

| Model                  | Purpose                            |
| ---------------------- | ---------------------------------- |
| Popularity             | Baseline / cold-start              |
| Transaction Popularity | Stronger purchase-based baseline   |
| User-User CF           | Collaborative filtering experiment |
| Item-Item CF           | Candidate generation               |
| ALS                    | Matrix-factorization experiment    |
| SASRec                 | Sequential reranking               |

Item-Item CF was selected as the candidate-generation component, while SASRec was integrated as the sequential reranker.

ALS and User-User CF remain useful experimental comparisons rather than the final production recommendation path.

---

# Important Engineering Problems Solved

Building ShopSense involved several problems that affected the correctness of the system.

## 1. Data Leakage

### Problem

A simple train/test split allowed future interactions from the same user to remain in training.

### Solution

Implemented chronological per-user splitting with cutoff validation.

```text
User events
     ↓
Chronological ordering
     ↓
Per-user cutoff
     ↓
Train < cutoff < Evaluation
```

---

## 2. ALS Mapping / KeyError

The ALS experiment exposed inconsistencies between user/item mappings and matrix orientation.

The issue was addressed by explicitly validating:

* user dimensions
* item dimensions
* mapping consistency
* factor matrix dimensions

This prevented silent ID-to-factor mismatches.

---

## 3. Sparse Matrix Dtype Mismatch

The Item-Item CF backend produced:

```text
Buffer dtype mismatch,
expected 'double' but got 'float'
```

The interaction matrix was corrected to the expected `float64` representation and validated before model training.

---

## 4. `filter_already_liked_items` Padding

Evaluation initially showed unexpected item IDs because recommendation arrays can contain zero-score padding after filtering.

This highlighted an important distinction between:

```text
Recommendation output size
```

and:

```text
Number of valid recommendation candidates
```

The evaluation protocol was adjusted appropriately for the next-interaction task rather than blindly filtering seen items.

---

## 5. Trivial Baseline Discovery

The repeat-history baseline produced very strong recall.

Instead of treating this as proof that the model was excellent, the result was investigated and used to understand the dataset.

This changed the evaluation philosophy from:

```text
"High Recall = Good recommender"
```

to:

```text
"Recall + repeat behavior + discovery + diversity
= better understanding of recommendation quality"
```

---

## 6. SASRec Design

The SASRec implementation required careful handling of:

* causal masking
* sequence padding
* maximum sequence length
* negative sampling
* observed-item exclusion
* GPU/CPU execution
* insufficient user history

The model is therefore integrated with fallback behavior rather than assumed to work for every user.

---

# MLflow

ShopSense uses **MLflow** for experiment tracking and model management.

Current setup:

```text
MLflow
│
├── Experiment
│     └── shopsense-production
│
├── Parameters
│
├── Metrics
│
└── Registered Models
      └── shopsense-item-item-cf
```

The project uses an SQLite MLflow backend:

```text
sqlite:///mlflow.db
```

Model artifacts and experiment information can therefore be tracked separately from the application code.

This makes the recommendation pipeline easier to reproduce and manage than simply keeping an untracked `.pkl` file.

---

# Prediction Pipeline

The prediction layer loads the trained artifacts and connects them to the recommendation service.

Conceptually:

```text
                     PredictionPipeline
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
 Item-Item CF           SASRec             Popularity
        │                   │                   │
        └──────────────┬────┴───────────────────┘
                       │
                       ▼
                 Recommendation
                    Service
                       │
                       ▼
                   Top-N Items
```

The pipeline also loads the product catalog so that model item IDs can be converted into storefront-ready product information.

---

# Recommendation Service

`RecommendationService` acts as the application-level coordinator.

Its responsibilities include:

* Accepting a user ID
* Loading persistent history
* Determining whether personalization is possible
* Selecting the appropriate recommendation strategy
* Generating recommendations
* Recording interactions
* Connecting the prediction pipeline with the EventStore

The application therefore does not need to know the internal details of Item-Item CF, SASRec, or popularity modeling.

---

# Recommendation Decision Logic

The application follows this general decision process:

```text
                 Request Recommendations
                          │
                          ▼
                 Is user known?
                    /       \
                  No         Yes
                  │           │
                  ▼           ▼
             Popularity   Load History
                              │
                              ▼
                    Enough history?
                       /       \
                     No         Yes
                     │           │
                     ▼           ▼
                Popularity   Item-Item CF
                              │
                              ▼
                         Candidate Pool
                              │
                              ▼
                           SASRec
                              │
                              ▼
                      Personalized Top-N
```

This design makes the fallback behavior explicit and testable.

---

# Streamlit Storefront

ShopSense includes a Streamlit application that exposes the recommendation pipeline through a simple e-commerce interface.

The storefront allows a user to:

* Enter a user ID
* Request recommendations
* View recommended products
* View recommendation source
* Interact with products
* Record transactions
* Start a new application session
* Continue using persistent history

The UI also makes the recommendation strategy visible.

Examples:

```text
🔥 Popularity
```

for fallback recommendations and:

```text
🧠 Item-Item CF + SASRec
```

for personalized recommendations.

This makes it easier to understand what the system is doing during development and demonstration.

---

# Testing

ShopSense includes automated tests for the recommendation system.

The test suite covers scenarios including:

### Anonymous / Unknown User

```text
Unknown User
     ↓
Popularity
     ↓
5+ recommendations
```

### Low-History User

```text
Known User
     ↓
Insufficient history
     ↓
Popularity fallback
```

### Known User Personalization

```text
Known User
     ↓
Stored history
     ↓
Item-Item CF + SASRec
```

### Cross-Session Personalization

```text
Session 1
   ↓
User interacts with products
   ↓
History persisted

Session 2
   ↓
Same User ID
   ↓
Previous history recovered
   ↓
Personalized recommendations
```

Run the recommendation-system tests with:

```bash
python -m tests.test_recommendation_system
```

---

# Project Structure

```text
ShopSense/
│
├── app.py
│
├── datasets/
│   └── RetailRocket data
│
├── docs/
│   ├── Evaluation_Improvements_Log.md
│   └── SASRec_Implementation_Notes.md
│
├── notebooks/
│
├── src/
│   │
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── feature_engineering.py
│   │   ├── model_trainer.py
│   │   └── catalog_builder.py
│   │
│   ├── configuration/
│   │
│   ├── database/
│   │   ├── event_store.py
│   │   └── services/
│   │       ├── recommendation_service.py
│   │       └── session_manager.py
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │   ├── prediction_pipeline.py
│   │   └── recommendation_pipeline.py
│   │
│   ├── model/
│   │
│   ├── logger/
│   │
│   ├── exception/
│   │
│   └── utils/
│
├── tests/
│   └── test_recommendation_system.py
│
├── product_catalog.parquet
├── requirements.txt
├── .gitignore
└── README.md
```

> The exact source-file names may vary as the project evolves; the structure above describes the major application components and their responsibilities.

---

# Technology Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy
* SciPy
* PyArrow

### Machine Learning

* Scikit-learn
* Implicit
* Joblib

### Deep Learning

* PyTorch

### Recommendation

* Item-Item Collaborative Filtering
* SASRec
* Popularity-based recommendation

### Experiment Tracking

* MLflow

### Application

* Streamlit

### Storage

* SQLite
* Parquet

### Configuration

* YAML
* Pydantic
* Python environment variables

---

# Dataset

ShopSense uses the **RetailRocket recommender-system dataset**.

The dataset contains:

* User events
* Item interactions
* Item properties
* Category information

The project works with event types such as:

```text
view
addtocart
transaction
```

The dataset is interaction-heavy and strongly repeat-oriented, which is why the evaluation includes both repeat-history and discovery-oriented analysis.

> Dataset statistics can differ depending on preprocessing, filtering, train/test construction, and catalog-building stages. The production catalog generated by the current pipeline contains approximately 234,561 item rows.

---

# Local Setup

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd ShopSense
```

---

## 2. Create a virtual environment

### Windows

```powershell
python -m venv shope_sense
```

Activate it:

```powershell
.\shope_sense\Scripts\Activate.ps1
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run the recommendation tests

```bash
python -m tests.test_recommendation_system
```

---

## 5. Start the Streamlit application

```bash
streamlit run app.py
```

The application will open in the browser.

---

# Example Application Flow

### New User

```text
User ID: new_user_123

No usable history detected.

Recommendation Source:
🔥 Popularity
```

### Returning User

```text
User ID: existing_user_456

Persistent interaction history found.

Recommendation Source:
🧠 Item-Item CF + SASRec
```

### Interaction

```text
User
 ↓
Views product
 ↓
Event stored in SQLite
 ↓
Recommendation request
 ↓
Updated user history
 ↓
New recommendation list
```

---

# Reproducibility

The project separates:

```text
Data
   ↓
Feature Engineering
   ↓
Training
   ↓
Model Artifacts
   ↓
Prediction
   ↓
Recommendation Service
   ↓
Application
```

This makes it possible to work on individual stages without coupling the entire system together.

MLflow is used to track trained models and experiments, while the prediction layer loads the required artifacts for inference.

---

# Honest Scope

ShopSense is a **portfolio and learning project**, not a production-scale commercial recommendation platform.

The project demonstrates:

* Recommendation-system fundamentals
* Collaborative filtering
* Sequential recommendation
* Cold-start handling
* Persistent personalization
* Model evaluation
* MLflow
* Application architecture
* Streamlit
* Automated testing

It does **not** currently claim:

* Real-time distributed recommendation serving
* Production-scale cloud deployment
* Online model retraining
* A/B testing with real customers
* Industrial-scale feature stores
* Guaranteed business conversion improvement

The evaluation results are specific to the RetailRocket dataset and the implemented evaluation protocol.

---

# What I Learned

Building ShopSense went beyond training a recommendation model.

The project exposed several practical lessons:

### 1. Evaluation can be misleading

A high Recall@K score is not enough to conclude that a recommender is useful.

Understanding the target distribution is equally important.

### 2. Data leakage can invalidate recommendation experiments

Recommendation systems are temporal by nature, so train/test construction must respect time.

### 3. Baselines matter

A simple repeat-history strategy revealed that the dataset contains a large amount of repeat behavior.

### 4. Different models solve different problems

Popularity, collaborative filtering, and sequential models should not necessarily compete for the same responsibility.

### 5. A model is only one part of a recommendation system

A usable recommender also requires:

```text
Data
+
Models
+
Evaluation
+
Persistence
+
Prediction Pipeline
+
Business/Application Logic
+
User Interface
```

---

# Future Improvements

Potential next steps include:

* Better discovery-oriented evaluation
* More sophisticated candidate generation
* Transformer-based ranking improvements
* Real-time event ingestion
* Online model updates
* Feature store integration
* FastAPI recommendation API
* Dockerized deployment
* Cloud deployment
* Recommendation monitoring
* A/B testing
* Vector-based product similarity
* Retrieval-Augmented Generation for product explanations
* LLM-powered recommendation explanations

---

# Project Goal

The goal of ShopSense is not simply to achieve a high recommendation metric.

The project aims to demonstrate how a recommendation model can be transformed into an **end-to-end machine-learning application**:

```text
Raw E-Commerce Events
        ↓
Data Processing
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Evaluation
        ↓
MLflow Model Management
        ↓
Prediction Pipeline
        ↓
Persistent User History
        ↓
Recommendation Service
        ↓
Streamlit Storefront
        ↓
User Interaction
        ↓
Updated History
        ↓
Personalized Recommendations
```

ShopSense therefore combines **machine learning, recommendation systems, software engineering, model management, persistence, testing, and application development** into one project.
