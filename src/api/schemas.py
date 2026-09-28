from pydantic import BaseModel,Field
from typing import Optional


class RecommendationRequest(BaseModel):
    user_id:Optional[int]=None
    n:int=Field(default=10,ge=1,le=100)


class Recommendation(BaseModel):
    rank:int
    model_item_idx:int
    retailrocket_item_id:int
    display_name:str
    category_id:Optional[int]=None
    available:Optional[int]=None
    recommendation_source:str

class RecommendationResponse(BaseModel):
    user_id:Optional[int]
    recommendations:list[Recommendation]

