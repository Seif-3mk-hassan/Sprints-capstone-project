from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ..rag_chain import RAGChain  
    from ..src.settings import get_settings  
except ImportError:  
    from src.rag_chain import RAGChain  
    from src.settings import get_settings  


app = FastAPI(
    title="SwiftDesk AI — RAG Support Agent",
    description="FastAPI backend exposing a Gemini-powered support reply generator (RAG optional).",
    version="1.0.0",
)


class GenerateRequest(BaseModel):
    issue: str = Field(..., min_length=3, description="Customer issue / ticket text")
    use_rag: bool = Field(True, description="Whether to use Chroma retrieval")
    prompt_style: str = Field(
        "zero-shot",
        description="Prompt style: zero-shot | few-shot | reasoned",
    )
    k: int = Field(3, ge=1, le=10, description="Top-k retrieved documents")
    return_retrieval: bool = Field(False, description="Include retrieved context in response")


class GenerateResponse(BaseModel):
    reply: str
    use_rag: bool
    prompt_style: str
    k: int
    retrieved: list[dict] | None = None


@app.get("/health")
def health():
    settings = get_settings()
    chroma_exists = settings.chroma_persist_dir.exists()
    has_api_key = bool(settings.google_api_key)
    return {
        "status": "ok",
        "mock_mode": settings.mock_mode,
        "has_google_api_key": has_api_key,
        "chroma_dir_exists": chroma_exists,
        "chroma_persist_dir": str(settings.chroma_persist_dir),
        "chroma_collection": settings.chroma_collection,
        "default_top_k": settings.default_top_k,
        "model": settings.gemini_model,
    }



@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
        chain = RAGChain()
        payload = chain.generate(
            req.issue,
            use_rag=req.use_rag,
            prompt_style=req.prompt_style,
            k=req.k,
            include_retrieval=req.return_retrieval,
        )
        return payload
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
