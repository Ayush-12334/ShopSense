import os
import sys
import pickle
import torch

from src.logger import logging
from src.exception import CustomeException
from src.entity.config_entity import PredictionConfig
from src.components.sasrec import SASREC, SASRec
from src.pipeline.recommendation import RecommendationPipeline


class PredictionPipeline:
    def __init__(self, prediction_config: PredictionConfig = None):
        try:
            logging.info("Initializing Prediction Pipeline")

            self.config = prediction_config or PredictionConfig.from_latest_run()
            self._validate_artifacts_exist()

            self.feature_artifacts = self._load_pickle(
                self.config.feature_artifacts_path,
                "FeatureArtifacts"
            )

            self.item_model = self._load_pickle(
                self.config.item_item_model_path,
                "Item-Item CF model"
            )

            self.popularity_model = self._load_pickle(
                self.config.popularity_model_path,
                "Popularity model"
            )

            self.last_session_by_user = self._load_pickle(
                self.config.last_session_path,
                "Last session by user"
            )

            self.sasrec, self.sasrec_model = self._load_sasrec()

            self.recommendation_pipeline = RecommendationPipeline(
                feature_artifacts=self.feature_artifacts,
                item_model=self.item_model,
                sasrec=self.sasrec,
                sasrec_model=self.sasrec_model,
                last_session_by_user=self.last_session_by_user,
                popularity_model=self.popularity_model,
                catalog_path=self.config.catalog_path,
                candidate_pool_size=self.config.candidate_pool_size,
                min_history=self.config.min_history,
                sasrec_max_seq_len=self.config.sasrec_max_seq_len
            )

            logging.info(
                f"Prediction Pipeline ready. "
                f"Loaded run: {self.config.run_dir}"
            )

        except Exception as e:
            raise CustomeException(e, sys) from e

    def _validate_artifacts_exist(self):
        try:
            required_paths = {
                "feature_artifacts_path": self.config.feature_artifacts_path,
                "item_item_model_path": self.config.item_item_model_path,
                "popularity_model_path": self.config.popularity_model_path,
                "last_session_path": self.config.last_session_path,
                "sasrec_model_path": self.config.sasrec_model_path,
                "catalog_path": self.config.catalog_path
            }

            missing = [
                name
                for name, path in required_paths.items()
                if not os.path.exists(path)
            ]

            if missing:
                raise FileNotFoundError(
                    f"Missing prediction artifacts: {missing}"
                )

            logging.info("All prediction artifacts found")

        except Exception as e:
            raise CustomeException(e, sys) from e

    def _load_pickle(self, path, label):
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)

            logging.info(
                f"{label} loaded from {path}"
            )

            return obj

        except Exception as e:
            raise CustomeException(e, sys) from e

    def _load_sasrec(self):
        try:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            sasrec = SASREC.__new__(SASREC)
            sasrec.config = self.config
            sasrec.device = device
            sasrec.model = None

            n_items = len(
                self.feature_artifacts.item_to_idx
            )

            sasrec_model = SASRec(
                n_items=n_items,
                max_len=self.config.sasrec_max_seq_len,
                d_model=self.config.sasrec_d_model,
                n_heads=self.config.sasrec_n_heads,
                n_layers=self.config.sasrec_n_layers,
                dropout=self.config.sasrec_dropout
            ).to(device)

            state_dict = torch.load(
                self.config.sasrec_model_path,
                map_location=device
            )

            sasrec_model.load_state_dict(state_dict)
            sasrec_model.eval()

            logging.info(
                f"SASRec model loaded on {device}"
            )

            return sasrec, sasrec_model

        except Exception as e:
            raise CustomeException(e, sys) from e

    def predict(self, user_id=None, N=10):
        try:
            return self.recommendation_pipeline.recommend(
                user_id=user_id,
                N=N
            )

        except Exception as e:
            raise CustomeException(e, sys) from e
        
    def predict_similar(self, retailrocket_item_id, N=10):
        try:
            if N <= 0:
                raise ValueError("N must be greater than 0")

            retailrocket_item_id = int(retailrocket_item_id)

            item_to_idx = self.feature_artifacts.item_to_idx

            if retailrocket_item_id not in item_to_idx:
                logging.warning(
                    f"Item {retailrocket_item_id} not found in training vocabulary"
                )
                return []

            model_item_idx = item_to_idx[retailrocket_item_id]

            logging.info(
                f"SIMILAR PRODUCT | retailrocket_id={retailrocket_item_id} "
                f"| model_idx={model_item_idx}"
            )

            return self.recommendation_pipeline.recommend_similar(
                model_item_idx=model_item_idx,
                N=N
            )

        except Exception as e:
            raise CustomeException(e, sys) from e
    def predict_from_live_history(self,user_vector,item_sequence,N=10):

        try:
            return self.recommendation_pipeline.recommend_from_live_history(
                user_vector=user_vector,
                item_sequence=item_sequence,
                N=N
            )
        except Exception as e:

            raise CustomeException(e, sys) from e

if __name__ == "__main__":
    prediction_pipeline = PredictionPipeline()

    test_users = [
        ("Anonymous", None),
        ("Low history", 9),
        ("Personalized", 7)
    ]

    for name, user_id in test_users:
        print(f"\n{'=' * 60}")
        print(f"TEST: {name} | user_id={user_id}")
        print(f"{'=' * 60}")

        recommendations = prediction_pipeline.predict(
            user_id=user_id,
            N=10
        )

        for rec in recommendations:
            print(rec)
