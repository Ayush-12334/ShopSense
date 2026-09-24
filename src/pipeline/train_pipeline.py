import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomeException

from src.components.ingest import DataIngestion
from src.components.features import FeatureEngineering
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import (
    DataIngestionConfig,
    FeatureEngineeringConfig,
    ModelTrainerConfig
)


class TrainingPipeline:

    def __init__(self):

        self.data_ingestion_config = DataIngestionConfig()
        self.feature_engineering_config = FeatureEngineeringConfig()
        self.model_trainer_config = ModelTrainerConfig()

    # ==========================================================
    # DATA INGESTION
    # ==========================================================

    def start_data_ingestion(self):

        try:

            logging.info("========== DATA INGESTION STARTED ==========")

            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            dataframe = data_ingestion.initiate_data_ingestion()

            ingestion_artifacts = (
                data_ingestion.initiate_train_test_split(dataframe)
            )

            logging.info("========== DATA INGESTION COMPLETED ==========")

            return ingestion_artifacts

        except Exception as e:
            raise CustomeException(e, sys) from e

    # ==========================================================
    # FEATURE ENGINEERING
    # ==========================================================

    def start_feature_engineering(self, ingestion_artifacts):

        try:

            logging.info("========== FEATURE ENGINEERING STARTED ==========")

            train_events = pd.read_csv(
                ingestion_artifacts.train_file_path
            )

            feature_engineering = FeatureEngineering(
                feature_engineering_config=self.feature_engineering_config
            )

            feature_artifacts = (
                feature_engineering.initiate_feature_engineering(
                    train_events
                )
            )


            logging.info(
                "========== FEATURE ENGINEERING COMPLETED =========="
            )

            return feature_artifacts

        except Exception as e:
            raise CustomeException(e, sys) from e

    # ==========================================================
    # MODEL TRAINING
    # ==========================================================

    def start_model_training(
        self,
        feature_artifacts,
        ingestion_artifacts
    ):

        try:

            logging.info("========== MODEL TRAINING STARTED ==========")

            train_events = pd.read_csv(
                    ingestion_artifacts.train_file_path
            )
            
            test_df = pd.read_csv(
                ingestion_artifacts.test_file_path
            )

            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config
            )

            # This internally trains:
            # 1. Item-Item CF
            # 2. SASRec
            # 3. Saves both models
            model_artifacts = model_trainer.initiate_model_training(
                feature_artifacts=feature_artifacts,
                test_df=test_df,
                train_events=train_events
            )   
            # ==========================================
            # SHOW MODEL RESULTS
            # ==========================================

            print("\n========================================")
            print("MODEL TRAINING RESULTS")
            print("========================================")

            print("\nItem-Item CF Metrics:")
            print(model_artifacts.item_item_metrics)

            print("\nCF + SASRec Metrics:")
            print(model_artifacts.sasrec_metrics)

            print("\nItem-Item Model:")
            print(model_artifacts.item_item_model_path)

            print("\nSASRec Model:")
            print(model_artifacts.sasrec_model_path)

      

            logging.info("========== MODEL TRAINING COMPLETED ==========")

            return model_artifacts

        except Exception as e:
            raise CustomeException(e, sys) from e

    # ==========================================================
    # RUN COMPLETE PIPELINE
    # ==========================================================

    def run_pipeline(self):

        try:

            # 1. Data Ingestion
            ingestion_artifacts = self.start_data_ingestion()

            # 2. Feature Engineering
            feature_artifacts = self.start_feature_engineering(
                ingestion_artifacts
            )

            # 3. Model Training
            model_artifacts = self.start_model_training(
                feature_artifacts,
                ingestion_artifacts
            )

            logging.info("========== PIPELINE COMPLETED ==========")

            return (
                ingestion_artifacts,
                feature_artifacts,
                model_artifacts
            )

        except Exception as e:
            raise CustomeException(e, sys) from e


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":

    pipeline = TrainingPipeline()

    (
        ingestion_artifacts,
        feature_artifacts,
        model_artifacts
    ) = pipeline.run_pipeline()

    print("\nPipeline completed successfully.")

    print(
        "\nTrain file:",
        ingestion_artifacts.train_file_path
    )

    print(
        "Test file:",
        ingestion_artifacts.test_file_path
    )

    print(
        "\nInteraction shape:",
        feature_artifacts.interaction_df.shape
    )

    print(
        "User-item matrix:",
        feature_artifacts.user_item_matrix.shape
    )

    print(
        "Item-user matrix:",
        feature_artifacts.item_user_matrix.shape
    )

    print("\nItem-Item model:")
    print(model_artifacts.item_item_model_path)

    print("\nSASRec model:")
    print(model_artifacts.sasrec_model_path)

    print("\nModel training completed.")

    print("\nPopularity Model:")
    print(model_artifacts.popularity_model_path)