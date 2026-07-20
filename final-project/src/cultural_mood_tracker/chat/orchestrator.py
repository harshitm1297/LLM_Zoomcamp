from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from cultural_mood_tracker.config import load_settings
from cultural_mood_tracker.rag.chroma_ingest import (
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_CHROMA_DB_DIR,
)
from cultural_mood_tracker.rag.embeddings import DEFAULT_EMBEDDING_MODEL
from cultural_mood_tracker.rag.llm import DEFAULT_MODEL, generate_answer
from cultural_mood_tracker.rag.prompting import (
    DEFAULT_MAX_CONTEXT_CHARS,
    build_prompt,
    system_instruction_for_variant,
)
from cultural_mood_tracker.rag.retrieval import RetrievedChunk, open_collection
from cultural_mood_tracker.rag.retriever import ApplicationRetriever

from .schemas import ChatResponse, OrchestratorResult


LOGGER = logging.getLogger(__name__)
RAG_MAX_NEW_TOKENS = 450


class ChatOrchestrator:
    """Single-path RAG chatbot backed by the local Chroma collection."""

    def __init__(
        self,
        *,
        persist_dir: Path = Path(DEFAULT_CHROMA_DB_DIR),
        collection_name: str = DEFAULT_CHROMA_COLLECTION,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        llm_model_name: str = DEFAULT_MODEL,
        top_k: int = 5,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        retriever: ApplicationRetriever | Any | None = None,
    ) -> None:
        settings = load_settings()
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.top_k = max(1, top_k)
        self.max_context_chars = max_context_chars
        self.prompt_variant = settings.prompt_variant
        self.llm_temperature = settings.llm_temperature
        self.retriever = retriever or ApplicationRetriever(
            strategy="vector",
            persist_dir=persist_dir,
            collection_name=collection_name,
            model_name=embedding_model_name,
            enable_query_rewriting=settings.enable_query_rewriting,
        )

    def answer(self, query: str) -> OrchestratorResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        started_at = time.perf_counter()
        chunks: list[RetrievedChunk] = self.retriever.retrieve(
            query.strip(),
            top_k=self.top_k,
            strategy="vector",
        )
        LOGGER.info("rag_retrieved_chunks=%s", len(chunks))
        prompt = build_prompt(
            query.strip(),
            chunks,
            max_context_chars=self.max_context_chars,
            system_instruction=system_instruction_for_variant(self.prompt_variant),
        )
        answer = generate_answer(
            prompt,
            self.llm_model_name,
            max_new_tokens=RAG_MAX_NEW_TOKENS,
            temperature=self.llm_temperature,
        )
        LOGGER.info("rag_total_latency=%.3fs", time.perf_counter() - started_at)
        return OrchestratorResult(answer=answer, retrieved_chunks=chunks)

    def healthcheck(self) -> dict[str, Any]:
        collection = open_collection(self.persist_dir, self.collection_name)
        chunk_count = int(collection.count())
        if chunk_count < 1:
            raise RuntimeError(f"Chroma collection {self.collection_name!r} is empty")
        return {
            "status": "healthy",
            "mode": "rag",
            "knowledge_base": "chroma",
            "collection": self.collection_name,
            "chunk_count": chunk_count,
            "retrieval_strategy": "vector",
        }


def answer_question(query: str) -> ChatResponse:
    return ChatOrchestrator().answer(query).to_response()
