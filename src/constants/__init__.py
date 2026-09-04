import os 



# ============================================================
# Pipeline
# ============================================================

PIPELINE_NAME:str='src'
ARTIFACT_DIR:str='artifacts'

# ============================================================
# Raw dataset
# ============================================================

FILE_NAME: str = "events.csv"

# ============================================================
# Data Ingestion
# ============================================================

DATA_INGESTION_DIR_NAME:str='data_ingestion'
DATA_INGESTION_FEATURE_STORE_DIR:str="feature_store"
DATA_INGESTION_INGESTED_DIR:str='ingested'
DATA_INGESTION_TRAIN_FILE_NAME: str = "train.csv"
DATA_INGESTION_TEST_FILE_NAME: str = "test.csv"

# ============================================================
# Data Split Configuration
# ============================================================

SPLIT_STRATEGY: str = "last_transaction_per_user"

# ============================================================
# Data Split Configuration
# ============================================================

EVENTS_WEIGHTS={
    'view':1,
    'addtocart':3,
    'transaction':5
}


FEATURE_ENGINEERING_DIR_NAME : str ="Feature_Engineering"
# ==============================
# MLflow
# ==============================

MLFLOW_EXPERIMENT_NAME = "shopsense-production"
ITEM_ITEM_REGISTERED_MODEL_NAME = "shopsense-item-item-cf"
SASREC_REGISTERED_MODEL_NAME = "shopsense-sasrec"
# ==============================
# Item-Item CF
# ==============================
ITEM_ITEM_K = 50
# ==============================
# SASRec
# ==============================
PAD=0
SESSION_GAP_MINUTES=30
SASREC_MAX_SEQ_LEN = 50
SASREC_D_MODEL = 64
SASREC_N_HEADS = 2
SASREC_N_LAYERS = 2
SASREC_N_EPOCHS = 5

