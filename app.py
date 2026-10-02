import re

import streamlit as st

from src.database.services.recommendation_service import (
    RecommendationService
)
from src.database.services.session_manager import (
    SessionManager
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ShopSense",
    page_icon="🛍️",
    layout="wide",
)


# ============================================================
# BACKEND
# ============================================================

@st.cache_resource
def get_service():
    return RecommendationService()


service = get_service()


# ============================================================
# SESSION STATE
# ============================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = (
        SessionManager.new_session_id()
    )

if "user_id" not in st.session_state:
    st.session_state.user_id = ""

if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

if "model_source" not in st.session_state:
    st.session_state.model_source = None


# ============================================================
# CLEAN PRODUCT NAME
# ============================================================

def clean_product_name(name):

    if name is None:
        return None

    name = str(name)

    # Remove HTML tags such as:
    # <div>, <span>, <p>, etc.
    name = re.sub(
        r"<[^>]+>",
        "",
        name
    )

    # Remove common HTML entities
    name = name.replace(
        "&nbsp;",
        " "
    )

    name = name.replace(
        "&amp;",
        "&"
    )

    name = name.replace(
        "&lt;",
        "<"
    )

    name = name.replace(
        "&gt;",
        ">"
    )

    # Remove extra whitespace
    name = " ".join(
        name.split()
    )

    return name.strip()


# ============================================================
# MODEL LABEL
# ============================================================

def get_model_name(source):

    source = str(
        source or ""
    ).lower()

    if source in {
        "popularity",
        "popularity_fallback",
    }:

        return "🔥 Popularity"

    if source in {
        "item_item_cf_sasrec",
        "cf_sasrec",
    }:

        return "🧠 Item-Item CF + SASRec"

    if source == "sasrec":

        return "🧠 SASRec"

    if source == "personalized_session":

        return "🧠 Session Personalization"

    if source == "similar_products":

        return "🔗 Similar Products"

    if source:

        return f"🧠 {source}"

    return "Recommendation Engine"


# ============================================================
# GET RECOMMENDATIONS
# ============================================================

def load_recommendations():

    user_id = (
        st.session_state.user_id.strip()
    )

    if not user_id:

        st.warning(
            "Please enter a User ID."
        )

        return

    try:

        recommendations = (
            service.get_recommendations(
                user_id=user_id,
                session_id=(
                    st.session_state.session_id
                ),
                N=10,
            )
        )

        st.session_state.recommendations = (
            recommendations or []
        )

        if recommendations:

            st.session_state.model_source = (
                recommendations[0].get(
                    "recommendation_source"
                )
            )

        else:

            st.session_state.model_source = None

    except Exception as e:

        st.error(
            "Unable to generate recommendations."
        )

        st.exception(e)


# ============================================================
# RECORD TRANSACTION
# ============================================================

def record_transaction(item_id):

    user_id = (
        st.session_state.user_id.strip()
    )

    if not user_id:

        st.warning(
            "Please enter a User ID."
        )

        return

    try:

        service.log_interaction(

            user_id=user_id,

            session_id=(
                st.session_state.session_id
            ),

            item_id=item_id,

            event_type="transaction",
        )

        st.success(
            f"Transaction recorded for item {item_id}."
        )

        # Refresh recommendations
        load_recommendations()

    except Exception as e:

        st.error(
            "Could not record transaction."
        )

        st.exception(e)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛍️ ShopSense"
)

st.caption(
    "Personalized Recommendation System"
)


# ============================================================
# USER INPUT
# ============================================================

st.subheader(
    "Enter User ID"
)

input_col, button_col = st.columns(
    [4, 1]
)


with input_col:

    st.session_state.user_id = st.text_input(

        "User ID",

        value=st.session_state.user_id,

        placeholder="Example: 7, 100, 999",

        label_visibility="collapsed",
    )


with button_col:

    if st.button(
        "Show Recommendations",
        type="primary",
        use_container_width=True,
    ):

        load_recommendations()


# ============================================================
# RECOMMENDATION RESULTS
# ============================================================

if st.session_state.recommendations:

    recommendations = (
        st.session_state.recommendations
    )

    source = (
        st.session_state.model_source
    )

    st.divider()

    # ========================================================
    # USER + MODEL
    # ========================================================

    user_col, model_col = st.columns(
        [1, 2]
    )

    with user_col:

        st.metric(
            "User",
            st.session_state.user_id
        )

    with model_col:

        st.metric(
            "Recommendation Model",
            get_model_name(source)
        )


    # ========================================================
    # MODEL DESCRIPTION
    # ========================================================

    if source in {
        "popularity",
        "popularity_fallback",
    }:

        st.info(
            "This user has little or no usable history, "
            "so ShopSense is using the Popularity model."
        )

    elif source in {
        "item_item_cf_sasrec",
        "cf_sasrec",
        "sasrec",
    }:

        st.success(
            "This user has existing behavioral history, "
            "so ShopSense is generating personalized recommendations."
        )


    # ========================================================
    # PRODUCTS
    # ========================================================

    st.subheader(
        "Recommended Products"
    )

    cols = st.columns(5)

    for i, recommendation in enumerate(
        recommendations[:10]
    ):

        with cols[i % 5]:

            # ------------------------------------------------
            # ITEM ID
            # ------------------------------------------------

            item_id = recommendation.get(
                "retailrocket_item_id"
            )

            if item_id is None:

                item_id = recommendation.get(
                    "item_id"
                )

            if item_id is None:

                item_id = recommendation.get(
                    "model_item_idx",
                    "Unknown"
                )


            # ------------------------------------------------
            # PRODUCT NAME
            # ------------------------------------------------

            name = recommendation.get(
                "display_name"
            )

            name = clean_product_name(
                name
            )

            if not name:

                name = (
                    f"Product {item_id}"
                )


            # ------------------------------------------------
            # CATEGORY
            # ------------------------------------------------

            category = recommendation.get(
                "category_id",
                "Unknown"
            )


            # ------------------------------------------------
            # PRODUCT CARD
            # ------------------------------------------------

            with st.container(
                border=True
            ):

                st.markdown(
                    f"### {name}"
                )

                st.caption(
                    f"Item ID: {item_id}"
                )

                st.caption(
                    f"Category: {category}"
                )


                # ------------------------------------------------
                # VIEW
                # ------------------------------------------------

                if st.button(
                    "View",
                    key=(
                        f"view_"
                        f"{item_id}_"
                        f"{i}"
                    ),
                    use_container_width=True,
                ):

                    try:

                        service.log_interaction(

                            user_id=(
                                st.session_state
                                .user_id
                                .strip()
                            ),

                            session_id=(
                                st.session_state
                                .session_id
                            ),

                            item_id=item_id,

                            event_type="view",
                        )

                        st.success(
                            "View recorded."
                        )

                    except Exception as e:

                        st.error(
                            "Could not record view."
                        )

                        st.exception(e)


                # ------------------------------------------------
                # TRANSACTION
                # ------------------------------------------------

                if st.button(
                    "🛒 Buy",
                    key=(
                        f"buy_"
                        f"{item_id}_"
                        f"{i}"
                    ),
                    use_container_width=True,
                ):

                    record_transaction(
                        item_id
                    )


    # ========================================================
    # NEW SESSION
    # ========================================================

    st.divider()

    if st.button(
        "Start New Session",
        use_container_width=False,
    ):

        st.session_state.session_id = (
            SessionManager.new_session_id()
        )

        # Keep the same User ID.
        #
        # This is important for testing
        # cross-session personalization.

        load_recommendations()

        st.rerun()


# ============================================================
# NO RESULTS / INITIAL SCREEN
# ============================================================

else:

    st.divider()

    st.info(
        "Enter a User ID and click "
        "'Show Recommendations'."
    )

    st.markdown(
        """
        **How ShopSense works**

        🆕 **New user** → Popularity recommendations

        👤 **Existing user** → Item-Item CF + SASRec

        The recommendation model is selected by the
        existing recommendation backend.
        """
    )