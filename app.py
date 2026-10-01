"""
app.py -- ShopSense storefront.

Reuses, never duplicates:
    RecommendationService -> all recommendations, similar products,
                             event logging
    PredictionPipeline    -> loaded once inside RecommendationService
    EventStore            -> accessed through RecommendationService
    CatalogBrowser        -> read-only browsing/search/catalog lookup
    SessionManager        -> session ID generation

Streamlit session_state contains UI state only.
Persistent behavioral history belongs to EventStore.
"""

import streamlit as st

from src.database.services.recommendation_service import RecommendationService
from src.database.catalog_browser import CatalogBrowser
from src.database.services.session_manager import SessionManager


# ======================================================================
# PAGE CONFIGURATION
# ======================================================================

st.set_page_config(
    page_title="ShopSense",
    page_icon="🛍️",
    layout="wide"
)


# ======================================================================
# BACKEND
# ======================================================================

@st.cache_resource
def load_service():
    return RecommendationService()


@st.cache_resource
def load_catalog():
    return CatalogBrowser()


try:
    service = load_service()
    catalog = load_catalog()
except Exception as backend_error:
    st.error(
        "ShopSense is temporarily unavailable. "
        "Please try again shortly."
    )

    with st.expander("Technical details"):
        st.exception(backend_error)

    st.stop()


# ======================================================================
# SESSION STATE
# ======================================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = SessionManager.new_session_id()

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "page" not in st.session_state:
    st.session_state.page = "home"

if "selected_item" not in st.session_state:
    st.session_state.selected_item = None

if "cart" not in st.session_state:
    st.session_state.cart = []

if "dev_mode" not in st.session_state:
    st.session_state.dev_mode = False

if "selected_category" not in st.session_state:
    st.session_state.selected_category = "All"

if "search_term" not in st.session_state:
    st.session_state.search_term = ""


# ======================================================================
# RECOMMENDATION SOURCE LABELS
# ======================================================================

SOURCE_LABELS = {
    "popularity": "Popular with other shoppers",
    "popularity_fallback": "Popular with other shoppers",
    "item_item_cf_sasrec": "Based on your shopping behavior",
    "similar_products": "Similar items",
    "personalized_session": "Based on what you viewed today",
}


def friendly_label(source):
    return SOURCE_LABELS.get(
        source,
        "Recommended for you"
    )


# ======================================================================
# SAFE BACKEND CALL
# ======================================================================

def safe_call(
    fn,
    default=None,
    error_label="Something went wrong"
):
    try:
        return fn()

    except Exception as e:
        st.error(
            f"{error_label}. Please try again."
        )

        if st.session_state.dev_mode:
            st.exception(e)

        return default


# ======================================================================
# NAVIGATION
# ======================================================================

def go_to_home():
    st.session_state.page = "home"
    st.rerun()


def go_to_product(item_id, log_view=True):
    st.session_state.selected_item = item_id
    st.session_state.page = "product"

    if log_view:
        safe_call(
            lambda: service.log_interaction(
                item_id=item_id,
                event_type="view",
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id
            ),
            error_label="Could not record this view"
        )

    st.rerun()


# ======================================================================
# SIDEBAR
# ======================================================================

with st.sidebar:

    st.title("🛍️ ShopSense")

    st.caption(
        "Shopping recommendations that adapt to your behavior"
    )

    st.divider()

    # ------------------------------------------------------------------
    # HOME
    # ------------------------------------------------------------------

    if st.button(
        "🏠 Home",
        use_container_width=True
    ):
        go_to_home()

    st.divider()

    # ------------------------------------------------------------------
    # ACCOUNT
    # ------------------------------------------------------------------

    st.subheader("Account")

    if st.session_state.user_id is not None:

        st.success(
            f"Signed in as **{st.session_state.user_id}**"
        )

        if st.button(
            "Sign out",
            use_container_width=True
        ):
            st.session_state.user_id = None
            st.session_state.session_id = (
                SessionManager.new_session_id()
            )
            st.session_state.cart = []
            st.session_state.page = "home"
            st.rerun()

    else:

        st.info("Browsing as guest")

        user_input = st.text_input(
            "Returning user?",
            placeholder="Enter User ID"
        )

        if st.button(
            "Sign in",
            use_container_width=True
        ):

            if user_input.strip():

                st.session_state.user_id = (
                    user_input.strip()
                )

                st.session_state.page = "home"

                st.rerun()

    # ------------------------------------------------------------------
    # SESSION
    # ------------------------------------------------------------------

    if st.button(
        "Start a new session",
        use_container_width=True
    ):

        st.session_state.session_id = (
            SessionManager.new_session_id()
        )

        st.session_state.cart = []

        st.session_state.selected_item = None

        st.session_state.page = "home"

        # IMPORTANT:
        # user_id is intentionally NOT cleared.
        # This allows the same user to return in a new session.

        st.rerun()

    st.divider()

    # ------------------------------------------------------------------
    # SHOPPING
    # ------------------------------------------------------------------

    st.subheader("Shopping")

    categories = safe_call(
        lambda: catalog.list_categories(),
        default=[],
        error_label="Could not load categories"
    )

    st.session_state.selected_category = st.selectbox(
        "Category",
        ["All"] + categories,
        index=(
            ["All"] + categories
        ).index(
            st.session_state.selected_category
        )
        if st.session_state.selected_category
        in ["All"] + categories
        else 0
    )

    st.session_state.search_term = st.text_input(
        "Search products",
        value=st.session_state.search_term,
        placeholder="Search by product name..."
    )

    st.caption(
        f"🛒 Cart: {len(st.session_state.cart)} item(s)"
    )

    st.divider()

    # ------------------------------------------------------------------
    # PERSONALIZATION
    # ------------------------------------------------------------------

    st.subheader("Personalization")

    if st.button(
        "✨ My Recommendations",
        use_container_width=True
    ):
        st.session_state.page = "recommendations"
        st.rerun()

    if st.button(
        "📊 My Activity",
        use_container_width=True
    ):
        st.session_state.page = "activity"
        st.rerun()

    st.divider()

    st.session_state.dev_mode = st.checkbox(
        "Developer / System Information"
    )


# ======================================================================
# PRODUCT CARD
# ======================================================================

def render_product_card(
    product,
    key_prefix
):
    item_id = product.get(
        "retailrocket_item_id"
    )

    display_name = product.get(
        "display_name",
        "Product"
    )

    category_id = product.get(
        "category_id",
        "—"
    )

    available = product.get(
        "available"
    )

    with st.container(border=True):

        st.markdown(
            f"### {display_name}"
        )

        st.caption(
            f"Category: {category_id}"
        )

        if available:
            st.success(
                "In stock",
                icon="✅"
            )
        else:
            st.warning(
                "Currently unavailable"
            )

        st.caption(
            f"Item ID: {item_id}"
        )

        if st.button(
            "View Product",
            key=f"{key_prefix}_view",
            use_container_width=True
        ):
            go_to_product(item_id)


# ======================================================================
# RECOMMENDATION CARD
# ======================================================================

def render_recommendation_row(
    items,
    exclude_id=None
):

    if exclude_id is not None:

        items = [
            item
            for item in items
            if item.get(
                "retailrocket_item_id"
            ) != exclude_id
        ]

    if not items:
        st.info(
            "Nothing to show here yet."
        )
        return

    columns = st.columns(
        min(len(items), 6)
    )

    for index, item in enumerate(
        items[:6]
    ):

        with columns[index]:

            item_id = item.get(
                "retailrocket_item_id"
            )

            st.markdown(
                f"**{item.get('display_name', 'Product')}**"
            )

            st.caption(
                f"Item ID: {item_id}"
            )

            if st.button(
                "View",
                key=(
                    f"recommendation_"
                    f"{item_id}_"
                    f"{index}_"
                    f"{st.session_state.page}"
                ),
                use_container_width=True
            ):
                go_to_product(item_id)

    if (
        items
        and st.session_state.dev_mode
    ):

        st.caption(
            "Recommendation source: "
            f"`{items[0].get('recommendation_source')}`"
        )


# ======================================================================
# PRODUCT ACTIONS
# ======================================================================

def render_product_actions(
    item_id,
    key_prefix
):

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------------
    # VIEW
    # --------------------------------------------------------------

    if col1.button(
        "View",
        key=f"{key_prefix}_view",
        use_container_width=True
    ):
        go_to_product(item_id)

    # --------------------------------------------------------------
    # ADD TO CART
    # --------------------------------------------------------------

    if col2.button(
        "Add to Cart",
        key=f"{key_prefix}_cart",
        use_container_width=True
    ):

        st.session_state.cart.append(
            item_id
        )

        success = safe_call(
            lambda: service.log_interaction(
                item_id=item_id,
                event_type="add_to_cart",
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id
            ),
            default=False,
            error_label="Could not record cart event"
        )

        if success is not False:
            st.success(
                "Added to cart"
            )

    # --------------------------------------------------------------
    # TRANSACTION
    # --------------------------------------------------------------

    if col3.button(
        "Buy",
        key=f"{key_prefix}_buy",
        use_container_width=True
    ):

        success = safe_call(
            lambda: service.log_interaction(
                item_id=item_id,
                event_type="transaction",
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id
            ),
            default=False,
            error_label="Could not record purchase"
        )

        if success is not False:
            st.success(
                "Purchase recorded"
            )


# ======================================================================
# HOME PAGE
# ======================================================================

def render_home():

    st.title(
        "Welcome to ShopSense 🛍️"
    )

    st.write(
        "Discover products and receive "
        "recommendations that adapt to your "
        "shopping behavior."
    )

    # --------------------------------------------------------------
    # PERSONALIZED SECTION
    # --------------------------------------------------------------

    if st.session_state.user_id is not None:

        st.subheader(
            "Recommended for You"
        )

        personalized = safe_call(
            lambda: service.get_recommendations(
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id,
                N=6
            ),
            default=[],
            error_label="Could not load recommendations"
        )

        if personalized:

            st.caption(
                friendly_label(
                    personalized[0].get(
                        "recommendation_source"
                    )
                )
            )

            render_recommendation_row(
                personalized
            )

        else:

            st.info(
                "Browse a few products to start "
                "building your personalized feed."
            )

    else:

        st.subheader(
            "Popular Products"
        )

        popular = safe_call(
            lambda: service.get_recommendations(
                user_id=None,
                session_id=None,
                N=6
            ),
            default=[],
            error_label="Could not load popular products"
        )

        render_recommendation_row(
            popular
        )

    st.divider()

    # --------------------------------------------------------------
    # PRODUCT BROWSING
    # --------------------------------------------------------------

    st.subheader(
        "Browse Products"
    )

    category_filter = (
        None
        if st.session_state.selected_category == "All"
        else st.session_state.selected_category
    )

    search_filter = (
        st.session_state.search_term
        if st.session_state.search_term.strip()
        else None
    )

    products = safe_call(
        lambda: catalog.list_products(
            limit=12,
            category_id=category_filter,
            search=search_filter
        ),
        default=[],
        error_label="Could not load products"
    )

    if not products:

        st.info(
            "No products match your search."
        )

    else:

        columns = st.columns(3)

        for index, product in enumerate(
            products
        ):

            with columns[index % 3]:

                render_product_card(
                    product,
                    key_prefix=(
                        f"home_"
                        f"{product.get('retailrocket_item_id')}"
                    )
                )


# ======================================================================
# PRODUCT DETAIL PAGE
# ======================================================================

def render_product():

    item_id = (
        st.session_state.selected_item
    )

    if item_id is None:

        st.info(
            "Select a product to see its details."
        )

        return

    product = safe_call(
        lambda: catalog.get_product(
            item_id
        ),
        default=None,
        error_label="Could not load this product"
    )

    if product is None:

        st.error(
            "This product could not be found."
        )

        return

    if st.button(
        "← Back to Store"
    ):

        st.session_state.page = "home"
        st.rerun()

    st.title(
        product.get(
            "display_name",
            "Product"
        )
    )

    st.write(
        f"**Category:** "
        f"{product.get('category_id', '—')}"
    )

    st.write(
        f"**Availability:** "
        f"{'In stock' if product.get('available') else 'Unavailable'}"
    )

    st.caption(
        f"Item ID: {item_id}"
    )

    st.divider()

    render_product_actions(
        item_id,
        key_prefix=f"detail_{item_id}"
    )

    st.divider()

    # --------------------------------------------------------------
    # SIMILAR PRODUCTS
    # --------------------------------------------------------------

    st.subheader(
        "Similar Products"
    )

    similar = safe_call(
        lambda: service.get_similar_products(
            item_id,
            N=6
        ),
        default=[],
        error_label="Could not load similar products"
    )

    render_recommendation_row(
        similar,
        exclude_id=item_id
    )


# ======================================================================
# RECOMMENDATIONS PAGE
# ======================================================================

def render_recommendations():

    st.title(
        "Recommended for You ✨"
    )

    if st.session_state.user_id is None:

        st.info(
            "Sign in as a returning user to "
            "see personalized recommendations."
        )

        return

    results = safe_call(
        lambda: service.get_recommendations(
            user_id=st.session_state.user_id,
            session_id=st.session_state.session_id,
            N=10
        ),
        default=[],
        error_label="Could not load your recommendations"
    )

    if not results:

        st.info(
            "Browse a few products and we'll "
            "start personalizing your recommendations."
        )

        return

    source = results[0].get(
        "recommendation_source"
    )

    st.caption(
        friendly_label(source)
    )

    render_recommendation_row(
        results
    )


# ======================================================================
# ACTIVITY PAGE
# ======================================================================

def render_activity():

    st.title(
        "Your Activity 📊"
    )

    if st.session_state.user_id is None:

        st.info(
            "Sign in to view your activity."
        )

        return

    events = safe_call(
        lambda: service.get_user_events(
            user_id=st.session_state.user_id,
            session_id=st.session_state.session_id,
            limit=50
        ),
        default=[],
        error_label="Could not load your activity"
    )

    if not events:

        st.info(
            "No activity recorded yet. "
            "Start browsing to build your profile."
        )

        return

    views = sum(
        1
        for event in events
        if event.get("event_type") == "view"
    )

    carts = sum(
        1
        for event in events
        if event.get("event_type") == "add_to_cart"
    )

    purchases = sum(
        1
        for event in events
        if event.get("event_type") == "transaction"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Views",
        views
    )

    col2.metric(
        "Add to Cart",
        carts
    )

    col3.metric(
        "Purchases",
        purchases
    )

    st.divider()

    st.subheader(
        "Personalization"
    )

    total_events = (
        views
        + carts
        + purchases
    )

    min_history = getattr(
        service,
        "min_session_history",
        0
    )

    if total_events < min_history:

        status = "New visitor"

    elif total_events < 5:

        status = "Building your profile"

    else:

        status = "Personalized"

    st.write(
        f"**Personalization status:** {status}"
    )

    st.divider()

    st.subheader(
        "Recommended Based on Your Activity"
    )

    recommendations = safe_call(
        lambda: service.get_recommendations(
            user_id=st.session_state.user_id,
            session_id=st.session_state.session_id,
            N=6
        ),
        default=[],
        error_label="Could not load recommendations"
    )

    render_recommendation_row(
        recommendations
    )


# ======================================================================
# PAGE ROUTER
# ======================================================================

PAGES = {
    "home": render_home,
    "product": render_product,
    "recommendations": render_recommendations,
    "activity": render_activity,
}

current_page = st.session_state.page

PAGES.get(
    current_page,
    render_home
)()


# ======================================================================
# DEVELOPER / SYSTEM INFORMATION
# ======================================================================

if st.session_state.dev_mode:

    with st.expander(
        "Developer / System Information",
        expanded=True
    ):

        st.write(
            "Session ID:",
            st.session_state.session_id
        )

        st.write(
            "User ID:",
            st.session_state.user_id
            or "(guest)"
        )

        history = safe_call(
            lambda: service.get_user_events(
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id
            ),
            default=[],
            error_label="Could not load history"
        )

        st.write(
            "Current-session interactions:",
            len(history)
        )

        st.write(
            "Minimum history for personalization:",
            getattr(
                service,
                "min_session_history",
                "Not exposed"
            )
        )