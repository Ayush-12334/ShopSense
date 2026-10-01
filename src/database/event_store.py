# ============================================================
# src/database/event_store.py
# ============================================================
# Persistent interaction storage for ShopSense.
#
# Responsibilities:
#   - Store user interactions
#   - Store session interactions
#   - Read interaction history
#   - Build sparse vectors for live recommendation
#
# This class does NOT:
#   - Train models
#   - Recommend products
#   - Modify SASRec
#   - Modify Item-Item CF
#   - Talk to MLflow
#
# Recommendation logic remains inside the recommendation
# / prediction pipeline.
# ============================================================

import sys
import sqlite3
import time

from scipy.sparse import csr_matrix

from src.logger import logging
from src.exception import CustomeException
from src.entity.config_entity import EventStoreConfig


class EventStore:

    def __init__(self, config=None):
        try:
            self.config = config if config is not None else EventStoreConfig()
            self.db_path = self.config.event_store_path
            self.event_weights = self.config.event_weights
            self.max_events_per_session = self.config.max_events_per_session
            self._init_db()

            logging.info(
                f"EventStore ready at {self.db_path}"
            )

        except Exception as e:
            raise CustomeException(e, sys) from e

    # ========================================================
    # DATABASE CONNECTION
    # ========================================================

    def _connect(self):
        return sqlite3.connect(self.db_path)

    # ========================================================
    # DATABASE INITIALIZATION
    # ========================================================

    def _init_db(self):
        try:
            with self._connect() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        session_id TEXT NOT NULL,
                        item_id INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        weight REAL NOT NULL,
                        timestamp INTEGER NOT NULL
                    )
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session
                    ON events(session_id)
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user
                    ON events(user_id)
                """)

        except Exception as e:
            raise CustomeException(e, sys) from e

    # ========================================================
    # WRITE EVENT
    # ========================================================

    def log_event(
        self,
        item_id,
        event_type,
        user_id=None,
        session_id=None,
        timestamp=None
    ):
        try:
            if session_id is None:
                raise ValueError("session_id is required")

            if event_type not in self.event_weights:
                raise ValueError(
                    f"Unknown event_type '{event_type}'. "
                    f"Expected: {list(self.event_weights.keys())}"
                )

            if item_id is None:
                raise ValueError("item_id is required")

            weight = self.event_weights[event_type]
            ts = timestamp if timestamp is not None else int(time.time())

            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO events
                    (
                        user_id,
                        session_id,
                        item_id,
                        event_type,
                        weight,
                        timestamp
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(user_id) if user_id is not None else None,
                        str(session_id),
                        int(item_id),
                        event_type,
                        weight,
                        ts
                    )
                )

            self._trim_session(session_id)

            logging.info(
                f"Event logged | "
                f"session={session_id} | "
                f"user={user_id} | "
                f"item={item_id} | "
                f"type={event_type}"
            )

        except Exception as e:
            raise CustomeException(e, sys) from e

    # ========================================================
    # SESSION LIMIT
    # ========================================================

    def _trim_session(self, session_id):
        try:
            with self._connect() as conn:
                count = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM events
                    WHERE session_id = ?
                    """,
                    (session_id,)
                ).fetchone()[0]

                if count > self.max_events_per_session:
                    excess = count - self.max_events_per_session

                    conn.execute(
                        """
                        DELETE FROM events
                        WHERE id IN (
                            SELECT id
                            FROM events
                            WHERE session_id = ?
                            ORDER BY timestamp ASC
                            LIMIT ?
                        )
                        """,
                        (session_id, excess)
                    )

        except Exception as e:
            raise CustomeException(e, sys) from e

    # ========================================================
    # READ EVENTS
    # ========================================================

    def get_events(
        self,
        user_id=None,
        session_id=None,
        limit=None
    ):
        try:
            if user_id is None and session_id is None:
                raise ValueError(
                    "Provide at least one of user_id or session_id"
                )

            query = """
                SELECT
                    item_id,
                    event_type,
                    weight,
                    timestamp
                FROM events
                WHERE
            """

            conditions = []
            params = []

            if user_id is not None:
                conditions.append("user_id = ?")
                params.append(str(user_id))

            if session_id is not None:
                conditions.append("session_id = ?")
                params.append(str(session_id))

            query += " OR ".join(conditions)
            query += " ORDER BY timestamp DESC"

            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)

            with self._connect() as conn:
                rows = conn.execute(
                    query,
                    params
                ).fetchall()

            return [
                {
                    "item_id": row[0],
                    "event_type": row[1],
                    "weight": row[2],
                    "timestamp": row[3]
                }
                for row in rows
            ]

        except Exception as e:
            raise CustomeException(e, sys) from e

    # ========================================================
    # RECENT ITEMS
    # ========================================================

    def get_recent_items(
        self,
        user_id=None,
        session_id=None,
        limit=20
    ):
        try:
            events = self.get_events(
                user_id=user_id,
                session_id=session_id,
                limit=limit * 3
            )

            seen = set()
            ordered = []

            for event in events:
                item_id = event["item_id"]

                if item_id not in seen:
                    seen.add(item_id)
                    ordered.append(item_id)

                if len(ordered) >= limit:
                    break

            return ordered

        except Exception as e:
            raise CustomeException(e, sys) from e

    # ========================================================
    # BUILD LIVE USER VECTOR
    # ========================================================

    def build_sparse_vector(
        self,
        item_to_idx,
        n_items,
        user_id=None,
        session_id=None
    ):
        try:
            events = self.get_events(
                user_id=user_id,
                session_id=session_id
            )

            weights_by_idx = {}

            for event in events:
                item_id = event["item_id"]

                if item_id not in item_to_idx:
                    continue

                idx = item_to_idx[item_id]

                weights_by_idx[idx] = (
                    weights_by_idx.get(idx, 0.0)
                    + event["weight"]
                )

            if not weights_by_idx:
                return None

            cols = list(weights_by_idx.keys())
            data = list(weights_by_idx.values())
            rows = [0 for _ in cols]

            return csr_matrix(
                (data, (rows, cols)),
                shape=(1, n_items)
            )

        except Exception as e:
            raise CustomeException(e, sys) from e