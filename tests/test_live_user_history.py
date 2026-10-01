from src.database.services.recommendation_service import RecommendationService

service = RecommendationService()

user_id = "new_user_100"
session_id = service.get_or_create_session()

print("SESSION:", session_id)

service.log_interaction(
    item_id=72028,
    event_type="view",
    user_id=user_id,
    session_id=session_id
)

service.log_interaction(
    item_id=216305,
    event_type="add_to_cart",
    user_id=user_id,
    session_id=session_id
)

print("\nSTORED EVENTS:")
print(service.event_store.get_events(user_id=user_id))

print("\nRECOMMENDATIONS:")
recommendations = service.get_recommendations(
    user_id=user_id,
    session_id=session_id,
    N=5
)

for item in recommendations:
    print(item)

    print("\nNEW SESSION TEST:")

new_session_id = service.get_or_create_session()

print("OLD SESSION:", session_id)
print("NEW SESSION:", new_session_id)

recommendations = service.get_recommendations(
    user_id=user_id,
    session_id=new_session_id,
    N=5
)

for item in recommendations:
    print(item)