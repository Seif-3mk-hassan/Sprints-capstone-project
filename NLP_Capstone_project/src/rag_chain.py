from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from .prompts import format_context, render_prompt
from .settings import Settings, get_settings


@dataclass(frozen=True)
class RetrievedChunk:
    document: str
    reference_reply: str | None = None
    id: str | None = None
    distance: float | None = None


class RAGChain:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

        # Create Gemini clients lazily so MOCK_MODE and offline demos don’t
        # fail during import/initialization.
        self._embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
        self._llm: Optional[ChatGoogleGenerativeAI] = None

    def _get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        self._validate_api_key()
        if self._embeddings is None:
            self._embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        return self._embeddings

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        self._validate_api_key()
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(model=self.settings.gemini_model, temperature=0.2)
        return self._llm

    def _validate_api_key(self) -> None:
        if self.settings.mock_mode:
            return
        if not self.settings.google_api_key:
            raise RuntimeError(
                "Missing GOOGLE_API_KEY. Set it in NLP_Capstone_project/.env (or NLP_Capstone_project/src/.env) or your environment."
            )

    def retrieve(self, query: str, *, k: int) -> list[RetrievedChunk]:
        if self.settings.mock_mode:
            return []

        persist_dir = Path(self.settings.chroma_persist_dir)
        if not persist_dir.exists():
            return []

        self._validate_api_key()

        client = chromadb.PersistentClient(path=str(persist_dir))
        collection = client.get_or_create_collection(name=self.settings.chroma_collection)

        if collection.count() == 0:
            return []

        query_emb = self._get_embeddings().embed_query(query)

        # chromadb returns a QueryResult; keep it as Any to avoid fragile
        # dependency on chromadb’s typing internals.
        res: Any = collection.query(
            query_embeddings=[query_emb],
            n_results=max(1, int(k)),
            include=["documents", "metadatas", "distances"],
        )

        documents = (res.get("documents") or [[]])[0]
        metadatas = (res.get("metadatas") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for doc, meta, dist, doc_id in zip(documents, metadatas, distances, ids):
            ref_reply = None
            if isinstance(meta, dict):
                ref_reply = meta.get("reference_reply")
            chunks.append(
                RetrievedChunk(
                    document=str(doc),
                    reference_reply=str(ref_reply) if ref_reply is not None else None,
                    id=str(doc_id) if doc_id is not None else None,
                    distance=float(dist) if dist is not None else None,
                )
            )
        return chunks

    def generate(
        self,
        issue: str,
        *,
        use_rag: bool = True,
        prompt_style: str = "zero-shot",
        k: int | None = None,
        include_retrieval: bool = False,
    ) -> dict[str, Any]:
        self._validate_api_key()

        top_k = int(k) if k is not None else int(self.settings.default_top_k)
        retrieved: list[RetrievedChunk] = []
        if use_rag:
            retrieved = self.retrieve(issue, k=top_k)

        context_block = None
        if retrieved:
            context_block = format_context(
                [
                    {
                        "document": c.document,
                        "reference_reply": c.reference_reply,
                    }
                    for c in retrieved
                ]
            )

        if self.settings.mock_mode:
            reply_text = (
                "Thanks for reaching out — I can help with this. "
                "Please share any error message, when it started, and what troubleshooting you’ve tried so far. "
                "In the meantime, try restarting the affected app/service and confirming your network connection."
            )
        else:
            prompt = render_prompt(style=prompt_style, issue=issue, context_block=context_block)
            result = self._get_llm().invoke(prompt)
            reply_text = getattr(result, "content", str(result))

        payload: dict[str, Any] = {
            "reply": reply_text,
            "use_rag": bool(use_rag),
            "prompt_style": prompt_style,
            "k": top_k,
        }

        if include_retrieval:
            payload["retrieved"] = [
                {
                    "id": c.id,
                    "distance": c.distance,
                    "document": c.document,
                    "reference_reply": c.reference_reply,
                }
                for c in retrieved
            ]

        return payload
