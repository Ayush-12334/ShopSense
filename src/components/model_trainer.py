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
from src.models.item_item_mlflow import ItemItemModel
from src.components.popularity import train_popularity_model
from src.models.sasrec_mlflow import SASRecModel
from src.models.popularity_mlflow import PopularityModelWrapper


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


                recall_metrics, total_users = metrics

                item_model_path=self.save_item_item(
                       item_model
                    )

                mlflow.pyfunc.log_model(
                    name="item_item_model",
                    python_model=ItemItemModel(
                        n_items=feature_artifacts.user_item_matrix.shape[1]

                    ),
                    artifacts={
                        "item_item_model":item_model_path
                    },

                    registered_model_name=self.config.item_item_register_name,
                    code_paths=["src/models/item_item_mlflow.py"],
                    pip_requirements=[
                        "mlflow",
                        "scipy",
                        "numpy",
                        "pandas",
                        "implicit"
                    ]



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

    def train_sasrec(self, train_events, feature_artifacts):


        try:

            with mlflow.start_run(run_name="sasrec") as run:

                # --------------------------------------------------
                # 1. Parameters
                # --------------------------------------------------

                mlflow.log_params({
                    "max_seq_len": self.config.sasrec_max_seq_len,
                    "d_model": self.config.sasrec_d_model,
                    "n_heads": self.config.sasrec_n_heads,
                    "n_layers": self.config.sasrec_n_layers,
                    "dropout": self.config.sasrec_dropout,
                    "epochs": self.config.sasrec_n_epochs,
                    "batch_size": self.config.sasrec_batch_size,
                    "learning_rate": self.config.sasrec_learning_rate,
                    "session_gap_minutes":
                        self.config.session_gap_minutes,
                    "n_items":
                        len(feature_artifacts.item_to_idx)
                })

                # --------------------------------------------------
                # 2. Create SASRec object
                # --------------------------------------------------

                sasrec = SASREC(
                    model_trainer_config=self.config
                )

                # --------------------------------------------------
                # 3. Create sequences
                # --------------------------------------------------

                session_sequences = sasrec.create_sequence(
                    train_events,
                    feature_artifacts.item_to_idx
                )

                # --------------------------------------------------
                # 4. Train
                # --------------------------------------------------

                sasrec_model = sasrec.train(
                    session_sequences,
                    len(feature_artifacts.item_to_idx)
                )

                # --------------------------------------------------
                # 5. Save local model
                # --------------------------------------------------

                sasrec_model_path = self.save_sasrec(
                    sasrec_model
                )

                # --------------------------------------------------
                # 6. Build last sessions
                # --------------------------------------------------

                last_session_by_user = (
                    sasrec.build_last_session_by_user(
                        train_events,
                        feature_artifacts.item_to_idx
                    )
                )

                # --------------------------------------------------
                # 7. Log local model as artifact
                # --------------------------------------------------

                mlflow.log_artifact(
                    sasrec_model_path,
                    artifact_path="sasrec_weights"
                )

                # --------------------------------------------------
                # 8. Log feature mappings
                # --------------------------------------------------

                import pickle

                item_to_idx_path = "item_to_idx.pkl"

                with open(item_to_idx_path, "wb") as f:
                    pickle.dump(
                        feature_artifacts.item_to_idx,
                        f
                    )

                mlflow.log_artifact(
                    item_to_idx_path,
                    artifact_path="features"
                )

                # --------------------------------------------------
                # 9. Log last sessions
                # --------------------------------------------------

                last_session_path = "last_session_by_user.pkl"

                with open(last_session_path, "wb") as f:
                    pickle.dump(
                        last_session_by_user,
                        f
                    )

                mlflow.log_artifact(
                    last_session_path,
                    artifact_path="features"
                )

                # --------------------------------------------------
                # 10. MLflow PyFunc registration
                # --------------------------------------------------

                

                mlflow.pyfunc.log_model(
                    artifact_path="sasrec_model",

                    python_model=SASRecModel(
                        n_items=len(
                            feature_artifacts.item_to_idx
                        ),
                        max_len=self.config.sasrec_max_seq_len,
                        d_model=self.config.sasrec_d_model,
                        n_heads=self.config.sasrec_n_heads,
                        n_layers=self.config.sasrec_n_layers,
                        dropout=self.config.sasrec_dropout,

                        item_to_idx=
                            feature_artifacts.item_to_idx,

                        idx_to_item=
                            feature_artifacts.idx_to_item,

                        last_session_by_user=
                            last_session_by_user
                    ),

                    artifacts={
                        "sasrec_model":
                            sasrec_model_path
                    },

                    registered_model_name=
                        self.config.sasrec_registered_name,

                    code_paths=[
                        "src/components/sasrec.py",
                        "src/models/sasrec_mlflow.py"
                    ],

                    pip_requirements=[
                        "mlflow",
                        "torch",
                        "numpy",
                        "pandas"
                    ]
                )

                # --------------------------------------------------
                # 11. Return MLflow run information
                # --------------------------------------------------

                sasrec_run_id = run.info.run_id

                logging.info(
                    f"SASRec MLflow run ID: {sasrec_run_id}"
                )

                logging.info(
                    f"SASRec model path: {sasrec_model_path}"
                )

                return (
                    sasrec,
                    sasrec_model,
                    sasrec_model_path,
                    sasrec_run_id,
                    last_session_by_user
            )

        except Exception as e:
        

            raise CustomeException(e, sys) from e

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

    

    def train_popularity(self,train_events:pd.DataFrame,feature_artifacts:FeatureArtifacts):

        try:
            logging.info("Starting Popularity model training")

            with mlflow.start_run(run_name="popularity") as run:
                mlflow.log_param("halflife_days",self.config.popularity_halflife_days)
                mlflow.log_param("n_items",len(feature_artifacts.item_to_idx))


                popularity_model=train_popularity_model(train_events=train_events,item_to_idx=feature_artifacts.item_to_idx,halflife_days=self.config.popularity_halflife_days)

                logging.info("popularity model training completed")
                # --------------------------------------------------
                # 3. Save staging file
                # -------------------------------------------------

                os.makedirs(self.config.model_trainer_dir,exist_ok=True)
                popularity_model_path = self.config.popularity_model_path
                with open(popularity_model_path,"wb") as f:
                    pickle.dump(popularity_model,f)
                # --------------------------------------------------
                # 4. Register model in MLflow
                # --------------------------------------------------
                
                mlflow.pyfunc.log_model(name='popularity_model',python_model=PopularityModelWrapper(),

                                    artifacts={"popularity_model":popularity_model_path},
                                    registered_model_name=self.config.popularity_registry_name,
                                    code_paths=[
                                        "src/components/popularity.py",
                                        "src/models/popularity_mlflow.py",
                                        
                                    ],

                                    pip_requirements=[
                                        "mlflow",
                                        "numpy",
                                        "pandas"
                                    ]


                                        )
            popularity_run_id=(
                run.info.run_id
            )

            
            popularity_model_url=(
                f"runs:/{popularity_run_id}/popularity_model"
            )

            logging.info(f"popularity mlflow run ID:"
                            f"{popularity_run_id}")

            return(
                popularity_model,
                popularity_model_path,
                popularity_run_id,
                popularity_model_url
            )

        except Exception as e:
            raise CustomeException(e,sys)



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

            (
                sasrec,
                sasrec_model,
                sasrec_model_path,
                sasrec_run_id,
                last_session_by_user
            ) = self.train_sasrec(
                train_events=train_events,
                feature_artifacts=feature_artifacts
            )


            
           
            logging.info("SASrec training completed")


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

           
            (
                popularity_model,
                popularity_model_path,
                popularity_run_id,
                popularity_model_url
            ) = self.train_popularity(
                train_events=train_events,
                feature_artifacts=feature_artifacts
            )
                        

            
    
            model_artifacts = ModelTrainerArtifacts(

                item_item_model_path=item_item_model_path,
                item_item_model_url=f"runs:/{item_item_run_id}/item_item_model",
                item_item_run_id=item_item_run_id,
                item_item_metrics=item_item_metrics,
                

                sasrec_model_path=sasrec_model_path,
                sasrec_model_url=f"runs:/{sasrec_run_id}/sasrec_model",
                sasrec_run_id=sasrec_run_id,
                sasrec_metrics=sasrec_metrics,

                popularity_model_path=popularity_model_path,
                popularity_model_url=popularity_model_url,
                popularity_run_id=popularity_run_id
            )



           
            logging.info("model training completed")

            return model_artifacts

           

        # --------------------------------------------------
        # 3. Return trained model for now
        # --------------------------------------------------


        except Exception as e:

            raise CustomeException(e,sys) from e