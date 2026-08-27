import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomeException

from src.components.ingest import DataIngestion
from src.components.features import FeatureEngineering
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import *


class TrainingPipeline:

    def __init__(self):

        self.data_ingestion_config = (
            DataIngestionConfig()
        )

        self.feature_engineering_config = (
            FeatureEngineeringConfig()
        )

        self.model_trainer_config=(
            ModelTrainerConfig()
        )


    def start_data_ingestion(self):

        logging.info(
            "========== DATA INGESTION STARTED =========="
        )

        data_ingestion = DataIngestion(
            data_ingestion_config=(
                self.data_ingestion_config
            )
        )

        # Load raw dataset.
        dataframe = (
            data_ingestion
            .initiate_data_ingestion()
        )

        # Create leakage-safe train/test split.
        ingestion_artifacts = (
            data_ingestion
            .initiate_train_test_split(
                dataframe
            )
        )

        logging.info(
            "========== DATA INGESTION COMPLETED =========="
        )

        return ingestion_artifacts


    def start_feature_engineering(
        self,
        ingestion_artifacts
    ):

        logging.info(
            "========== FEATURE ENGINEERING STARTED =========="
        )

        # Load only the training data.
        train_events = pd.read_csv(
            ingestion_artifacts.train_file_path
        )

        feature_engineering = (
            FeatureEngineering(
                feature_engineering_config=(
                    self.feature_engineering_config
                )
            )
        )

        feature_artifacts = (
            feature_engineering
            .initiate_feature_engineering(
                train_events
            )
        )

        logging.info(
            "========== FEATURE ENGINEERING COMPLETED =========="
        )

        return feature_artifacts


    def Start_Model_training(self,feature_artifacts,ingestion_artifacts):


        logging.info("====== model Training started")


        test_df=pd.read_csv(ingestion_artifacts.test_file_path)

        model_trainer=ModelTrainer(
            model_trainer_config=self.model_trainer_config
        )

        model_artifacts=(model_trainer.initiate_model_training(feature_artifacts=feature_artifacts,test_df=test_df))


        logging.info(
            "========== MODEL TRAINING COMPLETED =========="
        )


        return model_artifacts


    def run_pipeline(self):

        try:

            # --------------------------------
            # STEP 1: INGESTION
            # --------------------------------

            ingestion_artifacts = (
                self.start_data_ingestion()
            )


            # --------------------------------
            # STEP 2: FEATURE ENGINEERING
            # --------------------------------

            feature_artifacts = (
                self.start_feature_engineering(
                    ingestion_artifacts
                )
            )

            model_artifacts=(
                self.Start_Model_training(feature_artifacts=feature_artifacts,ingestion_artifacts=ingestion_artifacts)
            )


            logging.info(
                "========== PIPELINE COMPLETED =========="
            )

            return (
                ingestion_artifacts,
                feature_artifacts,
                model_artifacts
            )

        except Exception as e:

            raise CustomeException(
                e,
                sys
            ) from e


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
    print(
        "\nModel training completed."
    )

