from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None
    gemini_model: str

    mock_mode: bool

    chroma_persist_dir: Path
    chroma_collection: str

    default_top_k: int


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_settings(*, dotenv_path: str | os.PathLike | None = None) -> Settings:
    """Load settings from env + optional .env file.

    Env vars used:
      - GOOGLE_API_KEY (required for embeddings/LLM)
      - GEMINI_MODEL (optional)
      - CHROMA_PERSIST_DIR (optional)
      - CHROMA_COLLECTION (optional)
      - TOP_K_DEFAULT (optional)
    """

    project_root = get_project_root()

    if dotenv_path is None:
        default_env = project_root / ".env"
        fallback_env = project_root / "src" / ".env"
        if default_env.exists():
            load_dotenv(default_env)
        elif fallback_env.exists():
            load_dotenv(fallback_env)
        else:
            load_dotenv()
    else:
        load_dotenv(dotenv_path)

    google_api_key = os.getenv("GOOGLE_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    mock_mode = os.getenv("MOCK_MODE", "false").strip().lower() in {"1", "true", "yes", "y"}

    chroma_persist_dir = Path(
        os.getenv("CHROMA_PERSIST_DIR", str(project_root / "data" / "chroma_db"))
    )
    chroma_collection = os.getenv("CHROMA_COLLECTION", "support_tickets")

    try:
        default_top_k = int(os.getenv("TOP_K_DEFAULT", "3"))
    except ValueError:
        default_top_k = 3

    return Settings(
        google_api_key=google_api_key,
        gemini_model=gemini_model,

        mock_mode=mock_mode,
        chroma_persist_dir=chroma_persist_dir,
        chroma_collection=chroma_collection,
        default_top_k=default_top_k,
    )

