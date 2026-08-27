import sys

import mlflow
import pandas as pd

from src.logger import logging
from src.exception import CustomeException

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    FeatureArtifacts,
    ModelTrainerArtifacts
)

from src.evaluation.evaluate import build_eval_set, evaluate


from src.models.model import train_item_item


class ModelTrainer:

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig = ModelTrainerConfig()
    ):

        self.config = model_trainer_config

        # Connect to the MLflow Tracking Server
        # mlflow.set_tracking_uri(
        #     self.config.mlflow_tracking_uri
        # )

        # Select the experiment
        # mlflow.set_experiment(
        #     self.config.mlflow_experiment_name
        # )

    # ==========================================================
    # ITEM-ITEM CF
    # ==========================================================

    def train_item_item(
        self,
        feature_artifacts: FeatureArtifacts,
        eval_user_idx,
        eval_targets
    ):

        try:

            logging.info(
                "Starting Item-Item CF training"
            )

            # --------------------------------------------------
            # Start one MLflow run
            # --------------------------------------------------

            with mlflow.start_run(
                run_name="item_item_cf"
            ) as run:

                # --------------------------------------------------
                # 1. Read hyperparameter
                # --------------------------------------------------

                k = self.config.item_item_k

                # --------------------------------------------------
                # 2. Log hyperparameter to MLflow
                # --------------------------------------------------

                mlflow.log_param(
                    "item_item_k",
                    k
                )

                # --------------------------------------------------
                # 3. Train Item-Item model
                # --------------------------------------------------

                item_model = train_item_item(
                    feature_artifacts.user_item_matrix,
                    k=k
                )

                logging.info(
                    "Item-Item CF training completed"
                )

                # --------------------------------------------------
                # 4. Recommendation function
                # --------------------------------------------------

                def item_item_fn(user_idx, N):

                    ranked_items, scores = (
                        item_model.recommend(
                            userid=user_idx,
                            user_items=(
                                feature_artifacts
                                .user_item_matrix[user_idx]
                            ),
                            N=N,
                            filter_already_liked_items=False
                        )
                    )

                    return list(ranked_items)

                # --------------------------------------------------
                # 5. Evaluate model
                # --------------------------------------------------

                metrics = evaluate(
                    item_item_fn,
                    eval_user_idx,
                    eval_targets,
                    feature_artifacts.idx_to_item
                )

                logging.info(
                    f"Item-Item metrics: {metrics}"
                )

                # --------------------------------------------------
                # 6. Log metrics to MLflow
                # --------------------------------------------------

                # evaluate() returns:
                #
                # (
                #     {
                #         "Recall@5": ...,
                #         "Recall@10": ...,
                #         ...
                #     },
                #     total
                # )
                #
                # Therefore unpack it first.

                recall_metrics, total_users = metrics

                mlflow.log_metrics(
                    {
                        key.replace("@", "_at_"): value
                        for key, value in recall_metrics.items()
                    }
                )

                mlflow.log_metric(
                    "evaluation_users",
                    total_users
                )

                # --------------------------------------------------
                # 7. Log run information
                # --------------------------------------------------

                logging.info(
                    f"MLflow run ID: {run.info.run_id}"
                )

                return (
                    run.info.run_id,
                    recall_metrics,
                    item_model
                )

        except Exception as e:

            raise CustomeException(
                e,
                sys
            ) from e

    # ==========================================================
    # INITIATE MODEL TRAINING
    # ==========================================================

    def initiate_model_training(
        self,
        feature_artifacts: FeatureArtifacts,
        test_df: pd.DataFrame
    ):

        try:
            
            logging.info(
                "Starting Model Training"
            )

            # --------------------------------------------------
            # 1. Build evaluation dataset
            # --------------------------------------------------

            (
                eval_user_idx,
                eval_targets,
                eval_already_seen
            ) = build_eval_set(
                test_df=test_df,
                user_to_idx=feature_artifacts.user_to_idx,
                item_to_idx=feature_artifacts.item_to_idx,
                user_item_matrix=feature_artifacts.user_item_matrix
            )

            logging.info(
                f"Evaluable test users: "
                f"{len(eval_user_idx):,}"
            )

            logging.info(
                f"Novel targets: "
                f"{(~eval_already_seen).sum():,}"
            )

        # --------------------------------------------------
        # 2. Train Item-Item CF
        # --------------------------------------------------

            (
                tem_item_run_id,
                item_item_metrics,
                item_model
            ) = self.train_item_item(
                feature_artifacts,
                eval_user_idx,
                eval_targets
            )

            logging.info(
                "Model training completed"
            )

        # --------------------------------------------------
        # 3. Return trained model for now
        # --------------------------------------------------

            return item_model

        except Exception as e:

            raise CustomeException(e,sys) from e