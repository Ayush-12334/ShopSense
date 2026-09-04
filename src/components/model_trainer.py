import sys
import os 
import mlflow
import pickle
import numpy as np
import pandas as pd
from src.logger import logging
import torch
from src.exception import CustomeException
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    FeatureArtifacts,
    ModelTrainerArtifacts
)
from src.evaluation.evaluate import build_eval_set, evaluate
from src.models.model import train_item_item
from src.components.sasrec import SASREC


class ModelTrainer:

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig = ModelTrainerConfig()
    ):

        self.config = model_trainer_config

        # Connect to the MLflow Tracking Server
        mlflow.set_tracking_uri(
            self.config.mlflow_tracking_url
        )

        # Select the experiment
        mlflow.set_experiment(
            self.config.mlflow_experiment_name
        )

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

                item_model_path=self.save_item_item(
                       item_model
                    )         
            


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

                mlflow.log_artifact(
                    item_model_path,
                    artifact_path="item_item_model"
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
                    item_model,
                    item_model_path
                )

        except Exception as e:

            raise CustomeException(
                e,
                sys
            ) from e





    def train_sasrec(self,train_events,feature_artifacts):

        try:
            logging.info("starting SASrec training")

            sasrec=SASREC(
                model_trainer_config=self.config
            )

            session_sequence=sasrec.create_sequence(train_events=train_events,item_to_idx=feature_artifacts.item_to_idx)

            n_items=len(feature_artifacts.item_to_idx)

            sasrec_model=sasrec.train(session_sequences=session_sequence,n_items=n_items)

            logging.info("SASrec training completed ")

            return sasrec,sasrec_model

        except Exception as e:
            raise CustomeException(e,sys) from e

    # ==================== SAVE ITEM_ITEM ====================

    def save_item_item(self,item_model):
        try:
            os.makedirs(self.config.model_trainer_dir,exist_ok=True)

            with open(self.config.item_item_model_path,"wb") as f:
                pickle.dump(item_model,f)
            logging.info(
                f"Item-Item model saved:"
                f"{self.config.item_item_model_path}"
                )

            return self.config.item_item_model_path
        except Exception as e:
            raise CustomeException(e,sys) from e

    # ==================== SAVE SASREC ====================
    
    def save_sasrec(self, sasrec_model):

        try:
            os.makedirs(
                self.config.model_trainer_dir,
                exist_ok=True
            )

            torch.save(
                sasrec_model.state_dict(),
                self.config.sasrec_model_path
            )

            logging.info(
                f"SASRec model saved: "
                f"{self.config.sasrec_model_path}"
            )

            return self.config.sasrec_model_path

        except Exception as e:
            raise CustomeException(e, sys) from e


    def build_last_sessions(self,sasrec,train_events,feature_artifacts):

        try:
            logging.info("Building last session for sasrec inference")
            last_session_by_user=sasrec.build_last_session_by_user(
                train_events=train_events,
                item_to_idx=feature_artifacts.item_to_idx
            )

            logging.info(
                f"last session created for"
                f"{len(last_session_by_user):,} users"

            )
            return last_session_by_user
        except Exception as e :
            raise CustomeException(e,sys) from e

    def cf_sasrec_rerank(self,user_idx,N,item_model,sasrec,sasrec_model,last_session_by_user,feature_artifacts,candidate_pool_size=200):
        try:
            logging.info(f'Running CF+ SASRec reranking for user {user_idx}'
                         )


            ranked_items,cf_scores=item_model.recommend(
                userid=user_idx,
                user_items=(
                    feature_artifacts.user_item_matrix[user_idx]
                ),
                N=candidate_pool_size,filter_already_liked_items=False
            )

            candidates=list(ranked_items)

            if len(candidates) ==0:
                return []

            # --------------------------------------------------
            # 2. Convert user index -> original visitor ID
            # --------------------------------------------------
            

            visitor_id=feature_artifacts.idx_to_user[
                user_idx
            ]

            scores=sasrec.sasrec_score_candidates(
                visitor_id=visitor_id,
                candidate_item_idx=candidates,
                last_session_by_user=last_session_by_user,
                sasrec_model=sasrec_model,
                device=sasrec.device,
                max_seq_len=self.config.sasrec_max_seq_len

            )
             # --------------------------------------------------
            # 4. User has no usable session
            # --------------------------------------------------
            
            if scores is None:
                return candidates[:N]


            order=np.argsort(-scores)

            return [candidates[i] for i in order[:N]]


        except  Exception as e :
            raise CustomeException(e,sys) from e

    def evaluate_cf_sasrec(self,item_model,sasrec,sasrec_model,last_session_by_user,feature_artifacts,eval_user_idx,eval_targets):  

        try:

            logging.info(
                "Evaluating Item-Item CF + SASRec"
            )

            def rerank_fn(user_idx, N):

                return self.cf_sasrec_rerank(
                    user_idx=user_idx,
                    N=N,
                    item_model=item_model,
                    sasrec=sasrec,
                    sasrec_model=sasrec_model,
                    last_session_by_user=last_session_by_user,
                    feature_artifacts=feature_artifacts,
                    candidate_pool_size=200
                )

            metrics, total_users = evaluate(
                rerank_fn,
                eval_user_idx,
                eval_targets,
                feature_artifacts.idx_to_item
            )

            logging.info(
                f"Item-Item CF + SASRec metrics: {metrics}"
            )

            return metrics, total_users

        except Exception as e:

            raise CustomeException(e,sys) from e
            
    # ==========================================================
    # INITIATE MODEL TRAINING
    # ==========================================================

    def initiate_model_training(
        self,
        feature_artifacts: FeatureArtifacts,
        test_df: pd.DataFrame,
        train_events:pd.DataFrame
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
                item_item_run_id,
                item_item_metrics,
                item_model,
                item_item_model_path
            ) = self.train_item_item(
                feature_artifacts,
                eval_user_idx,
                eval_targets
            )

            logging.info(
                "Item-Item Model training completed"
            )

            # item_item_model_path=self.save_item_item(
            #     item_model
            #     )
            

            sasrec,sasrec_model=self.train_sasrec(
                train_events=train_events,
                feature_artifacts=feature_artifacts
            )
           
            logging.info("SASrec training completed")

            sasrec_model_path = self.save_sasrec(
                sasrec_model
                        )
            last_session_by_user = self.build_last_sessions(

                sasrec=sasrec,
                train_events=train_events,
                feature_artifacts=feature_artifacts
                )
            # --------------------------------------------------
            # 6. Evaluate Item-Item CF + SASRec
            # --------------------------------------------------

            (
                sasrec_metrics,
                sasrec_eval_users
            ) = self.evaluate_cf_sasrec(
                item_model=item_model,
                sasrec=sasrec,
                sasrec_model=sasrec_model,
                last_session_by_user=last_session_by_user,
                feature_artifacts=feature_artifacts,
                eval_user_idx=eval_user_idx,
                eval_targets=eval_targets
            )

           
            # 6. Create model artifacts

            model_artifacts = ModelTrainerArtifacts(
                item_item_model_path=item_item_model_path,
                item_item_model_url=item_item_model_path,
                item_item_run_id=item_item_run_id,
                item_item_metrics=item_item_metrics,

                sasrec_model_path=sasrec_model_path,
                sasrec_model_url=sasrec_model_path,
                sasrec_run_id=None,
                sasrec_metrics=sasrec_metrics
            )

            logging.info("model training completed")

            return model_artifacts

           

        # --------------------------------------------------
        # 3. Return trained model for now
        # --------------------------------------------------


        except Exception as e:

            raise CustomeException(e,sys) from e