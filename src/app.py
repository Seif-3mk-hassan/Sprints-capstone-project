from fastapi import FastAPI, HTTPException, Depends
import sqlite3
import os
from models import ProductSentimentSummary
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv


# ─── checking header ──────────────────────────
#API_KEY = "sprints-secret-key-value"
load_dotenv()

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key


# ─── Check database exists ──────────────────────────
DB_PATH = "data/reviews_db.sqlite"

if not os.path.exists(DB_PATH):
    print("❌ Database not found! Run etl_pipeline.py first.")
    print("   python etl_pipeline.py")
    exit(1)

# app = FastAPI(
#     title="Analytical Insights via a Secured FastAPI",
#     description="REST API connected to our ETL pipeline output",
#     version="1.0.0"
# )

app = FastAPI(
    title="Reviews Sentiment API",
    description="API for analyzing product reviews and sentiment scores using rolling averages",
    version="1.0.0",
    docs_url="/",

)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Returns dict-like rows
    return conn

@app.get("/health", tags=["Health"],summary="To check Database connectivity")
def health_check():
    try:
        conn = get_db()
        conn.execute("SELECT 1")
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")


@app.get("/api/v1/sentiment/{product_id}",dependencies=[Depends(verify_api_key)], response_model=ProductSentimentSummary, tags=["Sentiment"],summary="Get Product Sentiment by product id")
def get_product_sentiment(product_id: int):
    conn=get_db()
    cursor = conn.execute("""
        SELECT 
            product_id,
            product_name,
            rating AS latest_sentiment_score,
            rolling_average_sentiment
        FROM product_rolling_sentiment
        WHERE product_id = ?
        ORDER BY date DESC
        LIMIT 1
    """, (product_id,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"No data found for product_id {product_id}")
    return dict(row)


# ═══════════════════════════════════════════════════════
#  Run the server
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 55)
    print("FastAPI — Risk Sentiment API")
    print("=" * 55)
    print(f"""
  🚀 Starting server...
  
  📖 Docs:    http://127.0.0.1:8000/   (Swagger UI)
  
  Try these URLs in your browser:
    http://127.0.0.1:8000/health
    http://127.0.0.1:8000/api/v1/sentiment/1
  
  Press Ctrl+C to stop the server
""")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
