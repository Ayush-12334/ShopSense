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


@dataclass
class EventStoreConfig:

    event_store_path: str = (
        "shopsense_events.db"
    )

    event_weights: dict = field(
        default_factory=lambda: {
            "view": 1,
            "click": 2,
            "add_to_cart": 3,
            "transaction": 5
        }
    )

    max_events_per_session: int = 200

    max_recommendations_per_request: int = 50

    min_session_history: int = 2


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


# ============================================================
# PREDICTION PIPELINE CONFIG
# ============================================================

@dataclass
class PredictionConfig:
    run_dir: str = None

    feature_artifacts_path: str = None
    item_item_model_path: str = None
    sasrec_model_path: str = None
    popularity_model_path: str = None
    last_session_path: str = None

    catalog_path: str = "product_catalog.parquet"

    candidate_pool_size: int = 200
    min_history: int = 2

    sasrec_max_seq_len: int = SASREC_MAX_SEQ_LEN
    sasrec_d_model: int = SASREC_D_MODEL
    sasrec_n_heads: int = SASREC_N_HEADS
    sasrec_n_layers: int = SASREC_N_LAYERS
    sasrec_dropout: float = 0.2

    @classmethod
    def from_latest_run(cls):
        artifacts_root = os.path.join(
            PIPELINE_NAME,
            ARTIFACT_DIR
        )

        if not os.path.exists(artifacts_root):
            raise FileNotFoundError(
                f"Artifacts directory not found: {artifacts_root}"
            )

        run_dirs = [
            os.path.join(artifacts_root, name)
            for name in os.listdir(artifacts_root)
            if os.path.isdir(
                os.path.join(artifacts_root, name)
            )
        ]

        if not run_dirs:
            raise FileNotFoundError(
                "No training runs found in artifacts directory."
            )

        run_dirs.sort(
            key=os.path.getmtime,
            reverse=True
        )

        required_files = [
            os.path.join(
                "model_trainer",
                "item_item_model.pkl"
            ),
            os.path.join(
                "model_trainer",
                "sasrec_model.pt"
            ),
            os.path.join(
                "model_trainer",
                "popularity_model.pkl"
            ),
            os.path.join(
                "model_trainer",
                "feature_artifacts.pkl"
            ),
            os.path.join(
                "model_trainer",
                "last_session_by_user.pkl"
            )
        ]

        selected_run = None

        for run_dir in run_dirs:
            if all(
                os.path.exists(
                    os.path.join(run_dir, file)
                )
                for file in required_files
            ):
                selected_run = run_dir
                break

        if selected_run is None:
            raise FileNotFoundError(
                "No complete training run found with "
                "all required prediction artifacts."
            )

        model_trainer_dir = os.path.join(
            selected_run,
            "model_trainer"
        )

        return cls(
            run_dir=selected_run,
            feature_artifacts_path=os.path.join(
                model_trainer_dir,
                "feature_artifacts.pkl"
            ),
            item_item_model_path=os.path.join(
                model_trainer_dir,
                "item_item_model.pkl"
            ),
            sasrec_model_path=os.path.join(
                model_trainer_dir,
                "sasrec_model.pt"
            ),
            popularity_model_path=os.path.join(
                model_trainer_dir,
                "popularity_model.pkl"
            ),
            last_session_path=os.path.join(
                model_trainer_dir,
                "last_session_by_user.pkl"
            )
        )
# @dataclass
# class EventStoreConfig:
#     event_store_path: str = "shopsense_events.db"
#     event_weights: dict = field(default_factory=lambda: {
#         "view": 1,
#         "click": 2,
#         "add_to_cart": 3,
#         "transaction": 5
#     })
#     max_events_per_session: int = 200
#     max_recommendations_per_request: int = 50
#     min_session_history: int = 2

# @dataclass
# class EventStoreConfig:

#     event_store_path: str = (
#         "shopsense_events.db"
#     )

#     event_weights: dict = field(
#         default_factory=lambda: {
#             "view": 1,
#             "click": 2,
#             "add_to_cart": 3,
#             "transaction": 5
#         }
#     )

#     max_events_per_session: int = 200

#     max_recommendations_per_request: int = 50

#     min_session_history: int = 2