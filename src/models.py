from pydantic import BaseModel

class ProductSentimentSummary(BaseModel):
    product_id: int
    product_name: str
    latest_sentiment_score: float
    rolling_average_sentiment: float