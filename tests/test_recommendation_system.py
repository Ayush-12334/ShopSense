import sys
import uuid

from src.entity.config_entity import EventStoreConfig
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.database.event_store import EventStore
from src.database.services.recommendation_service import RecommendationService


TEST_USER_ID = f"test_user_{uuid.uuid4().hex[:8]}"
TEST_SESSION_ID = f"test_session_{uuid.uuid4().hex[:8]}"
TEST_ITEM_1 = 72028
TEST_ITEM_2 = 216305
TEST_USER_KNOWN = 7
TEST_USER_LOW_HISTORY = 9


def print_test(name):
    print("\n" + "=" * 70)
    print(f"TEST: {name}")
    print("=" * 70)


def check(condition, message):
    if not condition:
        raise AssertionError(f"FAILED: {message}")

    print(f"[PASS] {message}")


def test_prediction_pipeline():
    print_test("Prediction Pipeline")

    pipeline = PredictionPipeline()

    check(
        pipeline is not None,
        "PredictionPipeline loaded successfully"
    )

    return pipeline


def test_anonymous_recommendations(pipeline):
    print_test("Anonymous Recommendations")

    recommendations = pipeline.predict(
        user_id=None,
        N=5
    )

    check(
        len(recommendations) == 5,
        "Anonymous user received 5 recommendations"
    )

    check(
        all(
            item["recommendation_source"] == "popularity"
            for item in recommendations
        ),
        "Anonymous recommendations use popularity"
    )

    return recommendations


def test_low_history_user(pipeline):
    print_test("Low History User")

    recommendations = pipeline.predict(
        user_id=TEST_USER_LOW_HISTORY,
        N=5
    )

    check(
        len(recommendations) == 5,
        "Low-history user received recommendations"
    )

    check(
        all(
            item["recommendation_source"] == "popularity"
            for item in recommendations
        ),
        "Low-history user falls back to popularity"
    )

    return recommendations


def test_known_user_personalization(pipeline):
    print_test("Known User Personalization")

    recommendations = pipeline.predict(
        user_id=TEST_USER_KNOWN,
        N=5
    )

    check(
        len(recommendations) == 5,
        "Known user received recommendations"
    )

    check(
        all(
            item["recommendation_source"]
            == "item_item_cf_sasrec"
            for item in recommendations
        ),
        "Known user uses Item-Item CF + SASRec"
    )

    return recommendations


def test_similar_products(pipeline):
    print_test("Similar Products")

    recommendations = pipeline.predict_similar(
        retailrocket_item_id=TEST_ITEM_1,
        N=5
    )

    check(
        len(recommendations) == 5,
        "Similar-product request returned 5 products"
    )

    check(
        all(
            item["recommendation_source"] == "similar_products"
            for item in recommendations
        ),
        "Similar products use correct recommendation source"
    )

    check(
        all(
            int(item["retailrocket_item_id"]) != TEST_ITEM_1
            for item in recommendations
        ),
        "Original product is excluded from similar products"
    )

    return recommendations


def create_service():
    config = EventStoreConfig(
        event_store_path="test_shopsense_events.db"
    )

    service = RecommendationService(
        event_store_config=config
    )

    return service


def test_event_store(service):
    print_test("EventStore")

    session_id = TEST_SESSION_ID

    service.log_interaction(
        item_id=TEST_ITEM_1,
        event_type="view",
        user_id=TEST_USER_ID,
        session_id=session_id
    )

    service.log_interaction(
        item_id=TEST_ITEM_2,
        event_type="add_to_cart",
        user_id=TEST_USER_ID,
        session_id=session_id
    )

    events = service.event_store.get_events(
        user_id=TEST_USER_ID
    )

    check(
        len(events) >= 2,
        "User events were stored"
    )

    check(
        any(
            event["item_id"] == TEST_ITEM_1
            and event["event_type"] == "view"
            for event in events
        ),
        "VIEW event was stored correctly"
    )

    check(
        any(
            event["item_id"] == TEST_ITEM_2
            and event["event_type"] == "add_to_cart"
            for event in events
        ),
        "ADD_TO_CART event was stored correctly"
    )


def test_session_history(service):
    print_test("Session History")

    recent_items = service.event_store.get_recent_items(
        session_id=TEST_SESSION_ID,
        limit=10
    )

    check(
        TEST_ITEM_2 in recent_items,
        "Recent session contains add-to-cart item"
    )

    check(
        TEST_ITEM_1 in recent_items,
        "Recent session contains viewed item"
    )


def test_live_personalization(service):
    print_test("Live Personalization")

    recommendations = service.get_recommendations(
        user_id=TEST_USER_ID,
        session_id=TEST_SESSION_ID,
        N=5
    )

    check(
        len(recommendations) == 5,
        "Live user received 5 recommendations"
    )

    check(
        recommendations[0]["recommendation_source"] == "item_item_cf_sasrec_live",
        "Stored user history is used for live personalization"
    )

    return recommendations


def test_cross_session_personalization(service):
    print_test("Cross-Session Personalization")

    new_session_id = service.get_or_create_session()

    check(
        new_session_id != TEST_SESSION_ID,
        "A new session ID was created"
    )

    recommendations = service.get_recommendations(
        user_id=TEST_USER_ID,
        session_id=new_session_id,
        N=5
    )

    check(
        len(recommendations) == 5,
        "Returning user received recommendations"
    )

    check(
    recommendations[0]["recommendation_source"] == "item_item_cf_sasrec_live",
    "Previous-session history is used in the new session"
    )

    print(f"[PASS] Old session: {TEST_SESSION_ID}")
    print(f"[PASS] New session: {new_session_id}")

    return recommendations


def main():
    print("\n")
    print("=" * 70)
    print("SHOP SENSE - RECOMMENDATION SYSTEM INTEGRATION TEST")
    print("=" * 70)

    try:
        pipeline = test_prediction_pipeline()

        test_anonymous_recommendations(
            pipeline
        )

        test_low_history_user(
            pipeline
        )

        test_known_user_personalization(
            pipeline
        )

        test_similar_products(
            pipeline
        )

        service = create_service()

        test_event_store(
            service
        )

        test_session_history(
            service
        )

        test_live_personalization(
            service
        )

        test_cross_session_personalization(
            service
        )

        print("\n")
        print("=" * 70)
        print("ALL TESTS PASSED")
        print("=" * 70)

    except Exception as e:
        print("\n")
        print("=" * 70)
        print("TEST SUITE FAILED")
        print("=" * 70)
        print(f"ERROR: {e}")
        print("=" * 70)

        raise


if __name__ == "__main__":
    main()