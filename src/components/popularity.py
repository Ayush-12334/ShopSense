import sys
import numpy as np
import pandas as pd
from src.entity.config_entity import ModelTrainerConfig
from src.logger import logging
from src.exception import CustomeException


# ==================================================================
# POPULARITY / TRENDING FALLBACK MODEL
# ==================================================================
#
# This model is used ONLY as a fallback for users who do not have
# enough history for Item-Item CF / SASRec.
#
# Ranking:
#   1. Prefer transaction events
#   2. Apply exponential recency decay
#   3. Rank items by trending score
#
# Half-life = 14 days by default.
# ==================================================================


class PopularityModel:

    def __init__(self, popular_item_idx):
        """
        Store item indices in descending popularity/trending order.
        """
        self.popular_item_idx = popular_item_idx

    def recommend(self, N=10):
        """
        Return top-N popular/trending items.
        """
        return self.popular_item_idx[:N]


# ==================================================================
# TRAIN POPULARITY MODEL
# ==================================================================

def train_popularity_model(
    train_events: pd.DataFrame,
    item_to_idx: dict,
    halflife_days: float = 14
) -> PopularityModel:

    try:

        logging.info(
            "========== POPULARITY MODEL TRAINING STARTED =========="
        )

        # --------------------------------------------------
        # 1. Prefer transaction events
        # --------------------------------------------------

        events = train_events[
            train_events["event"] == "transaction"
        ].copy()

        # If there are no transactions, use all events
        if events.empty:

            logging.info(
                "No transaction events found. "
                "Using all events for popularity fallback."
            )

            events = train_events.copy()

        # --------------------------------------------------
        # 2. Calculate recency-decayed weight
        # --------------------------------------------------

        if "timestamp" in events.columns:

            ts = pd.to_numeric(
                events["timestamp"],
                errors="coerce"
            )

            # Keep only rows with valid timestamps
            valid_timestamp_mask = ts.notna()

            events = events.loc[
                valid_timestamp_mask
            ].copy()

            ts = ts.loc[
                valid_timestamp_mask
            ]

            if not events.empty:

                # RetailRocket timestamps are normally
                # epoch milliseconds.
                if ts.max() > 1e12:
                    ts_seconds = ts / 1000.0
                else:
                    ts_seconds = ts

                # Most recent timestamp in training data
                now = ts_seconds.max()

                # Age of every event in days
                age_days = (
                    now - ts_seconds
                ) / 86400.0

                # Exponential decay
                events["_weight"] = np.power(
                    0.5,
                    age_days / halflife_days
                )

            else:

                logging.warning(
                    "No valid timestamps found. "
                    "Using plain frequency."
                )

                events = train_events.copy()

                events["_weight"] = 1.0

        else:

            logging.warning(
                "Timestamp column not found. "
                "Using plain frequency."
            )

            events["_weight"] = 1.0

        # --------------------------------------------------
        # 3. Calculate trending score for each item
        # --------------------------------------------------

        trending_score = (
            events
            .groupby("itemid")["_weight"]
            .sum()
            .sort_values(ascending=False)
        )

        # --------------------------------------------------
        # 4. Convert item IDs -> internal item indices
        # --------------------------------------------------

        popular_item_ids = trending_score.index.tolist()

        popular_item_idx = [
            item_to_idx[item_id]
            for item_id in popular_item_ids
            if item_id in item_to_idx
        ]

        logging.info(
            f"Popularity fallback ranked "
            f"{len(popular_item_idx):,} items"
        )

        logging.info(
            "========== POPULARITY MODEL TRAINING COMPLETED =========="
        )

        return PopularityModel(
            popular_item_idx
        )

    except Exception as e:

        raise CustomeException(e, sys) from e
