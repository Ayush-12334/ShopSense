import sys
from src.logger import logging
from src.exception import CustomeException
from src.database.event_store import EventStore
from src.database.services.session_manager import SessionManager
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.entity.config_entity import EventStoreConfig

class RecommendationService:
    def __init__(self, prediction_pipeline=None, event_store=None, event_store_config=None):
        try:
            logging.info("Initializing Recommendation Service")
            self.event_store_config = event_store_config or EventStoreConfig()
            self.prediction_pipeline = prediction_pipeline or PredictionPipeline()
            self.event_store = event_store or EventStore(self.event_store_config)
            self.min_session_history = self.event_store_config.min_session_history
            self.max_recommendations = self.event_store_config.max_recommendations_per_request
            logging.info("Recommendation Service ready")
        except Exception as e:
            raise CustomeException(e, sys) from e

    def get_or_create_session(self, existing_session_id=None):
        return SessionManager.get_or_create(existing_session_id)

    def log_interaction(self, item_id, event_type, user_id=None, session_id=None):
        try:
            self.event_store.log_event(
                item_id=item_id,
                event_type=event_type,
                user_id=user_id,
                session_id=session_id
            )
        except Exception as e:
            raise CustomeException(e, sys) from e

    def get_similar_products(self, retailrocket_item_id, N=10):
        try:
            N = min(N, self.max_recommendations)
            return self.prediction_pipeline.predict_similar(
                retailrocket_item_id=retailrocket_item_id,
                N=N
            )
        except Exception as e:
            raise CustomeException(e, sys) from e

    def _build_live_item_sequence(self, user_id=None, session_id=None):
        try:
            events = self.event_store.get_events(
                user_id=user_id,
                session_id=session_id
            )
            item_to_idx = self.prediction_pipeline.feature_artifacts.item_to_idx
            sequence = []

            # get_events() returns newest first, so reverse for chronological order
            for event in reversed(events):
                retailrocket_item_id = event["item_id"]

                if retailrocket_item_id not in item_to_idx:
                    continue

                model_item_idx = int(item_to_idx[retailrocket_item_id])

                if not sequence or sequence[-1] != model_item_idx:
                    sequence.append(model_item_idx)

            max_seq_len = self.prediction_pipeline.config.sasrec_max_seq_len
            sequence = sequence[-max_seq_len:]

            logging.info(f"LIVE SASREC: sequence length = {len(sequence)}")
            return sequence

        except Exception as e:
            raise CustomeException(e, sys) from e

    def get_recommendations(self, user_id=None, session_id=None, N=10):
        try:
            N = min(N, self.max_recommendations)

            if user_id is not None:
                logging.info(f"Generating live recommendations for user {user_id}")

                user_vector = self.event_store.build_sparse_vector(
                    item_to_idx=self.prediction_pipeline.feature_artifacts.item_to_idx,
                    n_items=len(self.prediction_pipeline.feature_artifacts.item_to_idx),
                    user_id=user_id
                )

                item_sequence = self._build_live_item_sequence(
                    user_id=user_id,
                    session_id=session_id
                )

                if user_vector is not None:
                    logging.info(
                        f"User {user_id} has stored interaction history. "
                        "Using live Item-Item CF + SASRec."
                    )

                    results = self.prediction_pipeline.predict_from_live_history(
                        user_vector=user_vector,
                        item_sequence=item_sequence,
                        N=N
                    )

                    if results:
                        return results

                logging.info(f"No stored history for user {user_id}. Using popularity.")

                return self.prediction_pipeline.predict(
                    user_id=None,
                    N=N
                )

            if session_id is not None:
                recent_items = self.event_store.get_recent_items(
                    session_id=session_id,
                    limit=self.min_session_history
                )

                if len(recent_items) >= self.min_session_history:
                    most_recent_item = recent_items[0]

                    logging.info(
                        f"Session personalization using item {most_recent_item}"
                    )

                    results = self.prediction_pipeline.predict_similar(
                        retailrocket_item_id=most_recent_item,
                        N=N
                    )

                    for result in results:
                        result["recommendation_source"] = "personalized_session"

                    if results:
                        return results

            logging.info("No sufficient user/session history. Using popularity.")

            return self.prediction_pipeline.predict(
                user_id=None,
                N=N
            )

        except Exception as e:
            raise CustomeException(e, sys) from e