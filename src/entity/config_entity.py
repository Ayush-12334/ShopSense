import os
from dataclasses import dataclass
from datetime import datetime
from src.constants import *
TIMESTAMP: str=datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name:str=PIPELINE_NAME
    artifact_dir:str=os.path.join(PIPELINE_NAME,ARTIFACT_DIR,TIMESTAMP)
    timestamp:str=TIMESTAMP

training_pipeline_config=TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
     # Base directory
    data_ingestion_dir:str=os.path.join(training_pipeline_config.artifact_dir,DATA_INGESTION_DIR_NAME)
    # Raw/feature-store copy
    feature_store_file_path:str=os.path.join(data_ingestion_dir,DATA_INGESTION_FEATURE_STORE_DIR,FILE_NAME)
    # Directory containing train/test data
    ingested_dir:str=os.path.join(data_ingestion_dir,DATA_INGESTION_INGESTED_DIR)
    # Training data
    training_file_path:str=os.path.join(ingested_dir,DATA_INGESTION_TRAIN_FILE_NAME)
    # Testing data
    testing_file_path:str=os.path.join(ingested_dir,DATA_INGESTION_TEST_FILE_NAME)

@dataclass
class FeatureEngineeringConfig:
    """
    Configuration required for the feature engineering component
    """
    feature_dir: str=os.path.join(training_pipeline_config.artifact_dir,FEATURE_ENGINEERING_DIR_NAME)

@dataclass
class ModelTrainerConfig:


    model_trainer_dir:str=os.path.join(training_pipeline_config.artifact_dir,"model_trainer")

#============================
# Ml flow
#============================
    mlflow_tracking_url: str = os.getenv("MLFLOW_TRACKING_URL")
    #mlflow_tracking_url:str


    mlflow_experiment_name:str=(
        MLFLOW_EXPERIMENT_NAME

    )

    item_item_register_name:str=(
        ITEM_ITEM_REGISTERED_MODEL_NAME
    )

    sasrec_registerd_name:str=(
        SASREC_REGISTERED_MODEL_NAME
    )

#============================
# Item-Item CF
#============================

    item_item_k: int = ITEM_ITEM_K

# ==============================
# SASRec
# ==============================

    sasrec_max_seq_len: int = SASREC_MAX_SEQ_LEN

    sasrec_d_model: int = SASREC_D_MODEL

    sasrec_n_heads: int = SASREC_N_HEADS

    sasrec_n_layers: int = SASREC_N_LAYERS

    sasrec_n_epochs: int = SASREC_N_EPOCHS
