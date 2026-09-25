import os
from dataclasses import dataclass,field
from datetime import datetime

from src.constants import *


TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


# ============================================================
# TRAINING PIPELINE CONFIG
# ============================================================

@dataclass
class TrainingPipelineConfig:

    pipeline_name: str = PIPELINE_NAME

    artifact_dir: str = os.path.join(
        PIPELINE_NAME,
        ARTIFACT_DIR,
        TIMESTAMP
    )

    timestamp: str = TIMESTAMP


training_pipeline_config = TrainingPipelineConfig()


# ============================================================
# DATA INGESTION CONFIG
# ============================================================

@dataclass
class DataIngestionConfig:

    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_INGESTION_DIR_NAME
    )

    feature_store_file_path: str = os.path.join(
        data_ingestion_dir,
        DATA_INGESTION_FEATURE_STORE_DIR,
        FILE_NAME
    )

    ingested_dir: str = os.path.join(
        data_ingestion_dir,
        DATA_INGESTION_INGESTED_DIR
    )

    training_file_path: str = os.path.join(
        ingested_dir,
        DATA_INGESTION_TRAIN_FILE_NAME
    )

    testing_file_path: str = os.path.join(
        ingested_dir,
        DATA_INGESTION_TEST_FILE_NAME
    )


# ============================================================
# FEATURE ENGINEERING CONFIG
# ============================================================

@dataclass
class FeatureEngineeringConfig:

    feature_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        FEATURE_ENGINEERING_DIR_NAME
    )


# ============================================================
# MODEL TRAINER CONFIG
# ============================================================

@dataclass
class ModelTrainerConfig:

    # --------------------------------------------------------
    # Base model trainer directory
    # --------------------------------------------------------

    model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        "model_trainer"
    )

    # --------------------------------------------------------
    # Item-Item CF model
    # --------------------------------------------------------

    item_item_model_path: str = os.path.join(
        model_trainer_dir,
        "item_item_model.pkl"
    )

    item_item_k: int = ITEM_ITEM_K

    # --------------------------------------------------------
    # SASRec model
    # --------------------------------------------------------

    sasrec_model_path: str = os.path.join(
        model_trainer_dir,
        "sasrec_model.pt"
    )

    sasrec_max_seq_len: int = SASREC_MAX_SEQ_LEN

    sasrec_d_model: int = SASREC_D_MODEL

    sasrec_n_heads: int = SASREC_N_HEADS

    sasrec_n_layers: int = SASREC_N_LAYERS

    sasrec_n_epochs: int = SASREC_N_EPOCHS

    session_gap_minutes: int = SESSION_GAP_MINUTES

    pad: int = PAD
    

    # --------------------------------------------------------
    # SASRec training
    # --------------------------------------------------------

    sasrec_learning_rate: float = 1e-3

    sasrec_batch_size: int = 128

    sasrec_dropout: float = 0.2

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    mlflow_tracking_url: str = os.getenv(
    "MLFLOW_TRACKING_URL",
    "http://127.0.0.1:5000"
    )

    mlflow_experiment_name: str = (
        MLFLOW_EXPERIMENT_NAME
    )

    item_item_register_name: str = (
        ITEM_ITEM_REGISTERED_MODEL_NAME
    )

    sasrec_registered_name: str = (
        SASREC_REGISTERED_MODEL_NAME
    )


    # =========================
    # Popularity / Cold Start
    # =========================

    popularity_halflife_days:float=14

    popularity_registry_name:str=(POPULARITY_REGISTERED_MODEL_NAME)

    
    popularity_model_path: str = os.path.join(
        model_trainer_dir,
        "popularity_model.pkl"
    )
 


@dataclass
class CatalogConfig:
    item_properties_paths: list = field(default_factory=lambda: [
        "datasets/item_properties_part1.csv",
        "datasets/item_properties_part2.csv"
    ])
    category_tree_path: str = "datasets/category_tree.csv"
    train_events_path: str = "src/artifacts/08_11_2026_15_06_34/data_ingestion/ingested/train.csv"
    item_to_idx_path: str = "item_to_idx.pkl"
    output_catalog_path: str = "product_catalog.parquet"
    max_category_depth: int = 20
   