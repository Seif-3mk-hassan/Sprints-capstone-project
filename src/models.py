from pydantic import BaseModel

class ProductSentimentSummary(BaseModel):
    product_id: int
    product_name: str
    latest_sentiment_score: int  
    rolling_average_sentiment: float

    class Config:
        from_attributes = True
