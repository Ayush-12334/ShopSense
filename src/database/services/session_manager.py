"""
src/services/session_manager.py

Distinguishes an application session (a browser tab's visit, however
short) from a registered user_id and from the ML/SASRec session
concept (30-minute-gap sequence boundaries used at training time).

This module only produces/validates a session_id string. It does not
decide personalization logic -- that's recommendation_service.py.
"""

import sys
import uuid

from src.logger import logging
from src.exception import CustomeException


class SessionManager:

    @staticmethod
    def new_session_id():
        try:
            session_id = str(uuid.uuid4())
            logging.info(f"New session created: {session_id}")
            return session_id
        except Exception as e:
            raise CustomeException(e, sys) from e

    @staticmethod
    def get_or_create(existing_session_id=None):
        """
        Streamlit usage: store the returned value in st.session_state
        and pass it back in on every call within the same browser tab.
        """
        try:
            if existing_session_id:
                return existing_session_id
            return SessionManager.new_session_id()
        except Exception as e:
            raise CustomeException(e, sys) from e
