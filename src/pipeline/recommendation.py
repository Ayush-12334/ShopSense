# 
import os
import sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomeException


class RecommendationPipeline:

    def __init__(self, feature_artifacts, item_model, sasrec, sasrec_model,
                 last_session_by_user, popularity_model,
                 catalog_path="product_catalog.parquet", candidate_pool_size=200,
                 min_history=1, sasrec_max_seq_len=50):
        try:
            logging.info("Initializing Recommendation Pipeline")
            self.feature_artifacts = feature_artifacts
            self.item_model = item_model
            self.sasrec = sasrec
            self.sasrec_model = sasrec_model
            self.last_session_by_user = last_session_by_user
            self.popularity_model = popularity_model
            self.candidate_pool_size = candidate_pool_size
            self.min_history = min_history
            self.sasrec_max_seq_len = sasrec_max_seq_len

            if not os.path.exists(catalog_path):
                raise FileNotFoundError(f"Product catalog not found: {catalog_path}")

            self.catalog = pd.read_parquet(catalog_path)
            required_columns = {"retailrocket_item_id", "model_item_idx", "display_name", "category_id", "available"}
            missing_columns = required_columns - set(self.catalog.columns)
            if missing_columns:
                raise ValueError(f"Product catalog is missing columns: {missing_columns}")

            self.catalog_by_model_idx = self.catalog.drop_duplicates(subset=["model_item_idx"]).set_index("model_item_idx")
            logging.info(f"Product catalog loaded: {len(self.catalog):,} rows")
            logging.info("Recommendation Pipeline initialized")
        except Exception as e:
            raise CustomeException(e, sys) from e

    def _get_user_idx(self, user_id):
        try:
            if user_id not in self.feature_artifacts.user_to_idx:
                return None
            return self.feature_artifacts.user_to_idx[user_id]
        except Exception as e:
            raise CustomeException(e, sys) from e

    def _get_history_count(self, user_idx):
        try:
            if user_idx is None:
                return 0
            return int(self.feature_artifacts.user_item_matrix[user_idx].getnnz())
        except Exception as e:
            raise CustomeException(e, sys) from e

    def _recommend_personalized(self, user_idx, N):
        try:
            logging.info(f"PERSONALIZED: generating Item-Item CF candidates for user idx {user_idx}")
            ranked_items, cf_scores = self.item_model.recommend(
                userid=user_idx,
                user_items=self.feature_artifacts.user_item_matrix[user_idx],
                N=self.candidate_pool_size,
                filter_already_liked_items=False
            )
            candidates = list(ranked_items)
            logging.info(f"PERSONALIZED: Item-Item CF candidates = {len(candidates)}")

            if not candidates:
                logging.warning("PERSONALIZED: Item-Item CF returned no candidates")
                return []

            visitor_id = self.feature_artifacts.idx_to_user[user_idx]
            scores = self.sasrec.sasrec_score_candidates(
                visitor_id=visitor_id,
                candidate_item_idx=candidates,
                last_session_by_user=self.last_session_by_user,
                sasrec_model=self.sasrec_model,
                device=self.sasrec.device,
                max_seq_len=self.sasrec_max_seq_len
            )

            if scores is None:
                logging.info(f"PERSONALIZED: SASRec returned None for visitor {visitor_id}. Using Item-Item CF ranking.")
                return candidates[:N]

            logging.info(f"PERSONALIZED: SASRec returned scores for {len(scores)} candidates. Reranking.")
            order = np.argsort(-scores)
            return [candidates[i] for i in order[:N]]
        except Exception as e:
            raise CustomeException(e, sys) from e

    def _recommend_popular(self, N):
        try:
            return list(self.popularity_model.recommend(N=N))
        except Exception as e:
            raise CustomeException(e, sys) from e

    def _popularity_response(self, N, source):
        logging.info(f"POPULARITY FALLBACK USED (source={source})")
        return self._attach_catalog_information(self._recommend_popular(N), source)

    def _attach_catalog_information(self, model_item_indices, recommendation_source):
        try:
            results = []
            for rank, model_item_idx in enumerate(model_item_indices, start=1):
                model_item_idx = int(model_item_idx)
                if model_item_idx not in self.catalog_by_model_idx.index:
                    logging.warning(f"Model item index {model_item_idx} not found in product catalog")
                    continue
                product = self.catalog_by_model_idx.loc[model_item_idx]
                results.append({
                    "rank": rank,
                    "model_item_idx": model_item_idx,
                    "retailrocket_item_id": product["retailrocket_item_id"],
                    "display_name": product["display_name"],
                    "category_id": product["category_id"],
                    "available": product["available"],
                    "recommendation_source": recommendation_source
                })
            return results
        except Exception as e:
            raise CustomeException(e, sys) from e

    def recommend(self, user_id=None, N=10):
        try:
            if N <= 0:
                raise ValueError("N must be greater than 0")

            if user_id is None:
                logging.info("No user ID provided.")
                return self._popularity_response(N, "popularity")

            user_idx = self._get_user_idx(user_id)
            logging.info(f"USER ID = {user_id} (type={type(user_id).__name__}), USER IDX = {user_idx}")

            if user_idx is None:
                logging.info(f"User {user_id} not found in training vocabulary.")
                return self._popularity_response(N, "popularity")

            history = self._get_history_count(user_idx)
            logging.info(f"USER {user_id} HISTORY COUNT = {history} (min_history = {self.min_history})")

            if history < self.min_history:
                logging.info(f"User {user_id} has insufficient history.")
                return self._popularity_response(N, "popularity")

            logging.info("ENTERING PERSONALIZED RECOMMENDATION")
            model_item_indices = self._recommend_personalized(user_idx, N)

            if not model_item_indices:
                logging.warning("Personalized recommendation returned nothing.")
                return self._popularity_response(N, "popularity_fallback")

            logging.info(f"PERSONALIZED RECOMMENDATIONS RETURNED = {len(model_item_indices)}")
            return self._attach_catalog_information(model_item_indices, "item_item_cf_sasrec")

        except Exception as e:
            raise CustomeException(e, sys) from e

    def print_recommendations(self, user_id=None, N=10):
        try:
            recommendations = self.recommend(user_id=user_id, N=N)
            print("\nSHOP SENSE RECOMMENDATIONS")
            print("=" * 50)
            for recommendation in recommendations:
                print(f"\n{recommendation['rank']}. {recommendation['display_name']}")
                print(f"   Item ID: {recommendation['retailrocket_item_id']}")
                print(f"   Category: {recommendation['category_id']}")
                print(f"   Available: {recommendation['available']}")
                print(f"   Source: {recommendation['recommendation_source']}")
            return recommendations
        except Exception as e:
            raise CustomeException(e, sys) from e