from fastapi import FastAPI, HTTPException
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.api.schemas import RecommendationRequest, RecommendationResponse


app=FastAPI(
    tile="Shopsense Recommendation API",
    description="recommendation API powered by Item-Item CF ,SASRec and popularity",
    version="1.0.0"
)


prediction_pipeline=PredictionPipeline()


@app.get("/health")
def health():
    return {"status":"healthy"}


@app.post("/Recommendations",response_model=RecommendationResponse)

def get_recommendation(request:RecommendationRequest):

    try:
        recommendations=prediction_pipeline.predict(user_id=request.user_id,N=request.n)
        return {
            "user_id":request.user_id,
            "recommendations":recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    