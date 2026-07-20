from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chroma_ingest import DEFAULT_CHROMA_COLLECTION, DEFAULT_CHROMA_DB_DIR
from .document_chunks import load_document_chunks
from .embeddings import DEFAULT_EMBEDDING_MODEL
from .retrieval import RetrievedChunk, open_collection, query_collection
from .query_rewriting import rewrite_query


SUPPORTED_RETRIEVAL_STRATEGIES = {"bm25", "vector", "vector_reranked", "hybrid"}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("metadata"), dict):
        return dict(row["metadata"])
    return {
        key: value
        for key, value in row.items()
        if key not in {"chunk_id", "chunk_text"} and value is not None
    }


@dataclass
class BM25Index:
    chunks: list[dict[str, Any]]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.documents = [tokenize(str(row.get("chunk_text") or "")) for row in self.chunks]
        self.term_frequencies = [Counter(document) for document in self.documents]
        self.document_lengths = [len(document) for document in self.documents]
        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths) if self.document_lengths else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for document in self.documents:
            document_frequency.update(set(document))
        count = len(self.documents)
        self.inverse_document_frequency = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_terms = tokenize(query)
        scored: list[tuple[float, int]] = []
        for index, frequencies in enumerate(self.term_frequencies):
            score = 0.0
            length = self.document_lengths[index]
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b
                    + self.b * length / max(self.average_document_length, 1.0)
                )
                score += self.inverse_document_frequency.get(term, 0.0) * (
                    frequency * (self.k1 + 1) / denominator
                )
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda item: (-item[0], str(self.chunks[item[1]].get("chunk_id"))))
        maximum = scored[0][0] if scored else 1.0
        results: list[RetrievedChunk] = []
        for score, index in scored[:top_k]:
            row = self.chunks[index]
            normalized_score = score / maximum if maximum else 0.0
            results.append(
                RetrievedChunk(
                    chunk_id=str(row["chunk_id"]),
                    chunk_text=str(row["chunk_text"]),
                    metadata=_metadata(row),
                    distance=1.0 - normalized_score,
                )
            )
        return results


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]],
    *,
    top_k: int,
    rank_constant: int = 60,
) -> list[RetrievedChunk]:
    scores: defaultdict[str, float] = defaultdict(float)
    chunks_by_id: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] += 1.0 / (rank_constant + rank)
            chunks_by_id.setdefault(chunk.chunk_id, chunk)

    ordered_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
    maximum = max((scores[chunk_id] for chunk_id in ordered_ids), default=1.0)
    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            chunk_text=chunks_by_id[chunk_id].chunk_text,
            metadata=chunks_by_id[chunk_id].metadata,
            distance=1.0 - scores[chunk_id] / maximum,
        )
        for chunk_id in ordered_ids
    ]


class ApplicationRetriever:
    """Lazy configurable retriever shared by the app and evaluation code."""

    def __init__(
        self,
        *,
        strategy: str = "vector_reranked",
        persist_dir: Path = Path(DEFAULT_CHROMA_DB_DIR),
        collection_name: str = DEFAULT_CHROMA_COLLECTION,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        candidate_k: int = 20,
        chunks: list[dict[str, Any]] | None = None,
        collection: Any | None = None,
        enable_query_rewriting: bool = True,
    ) -> None:
        normalized = strategy.strip().lower()
        if normalized not in SUPPORTED_RETRIEVAL_STRATEGIES:
            choices = ", ".join(sorted(SUPPORTED_RETRIEVAL_STRATEGIES))
            raise ValueError(f"Unknown retrieval strategy {strategy!r}; expected one of {choices}")
        self.strategy = normalized
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.model_name = model_name
        self.candidate_k = max(1, candidate_k)
        self._chunks = chunks
        self._collection = collection
        self.enable_query_rewriting = enable_query_rewriting
        self._bm25: BM25Index | None = BM25Index(chunks) if chunks is not None else None

    def _lexical_index(self) -> BM25Index:
        if self._bm25 is None:
            self._chunks = load_document_chunks()
            self._bm25 = BM25Index(self._chunks)
        return self._bm25

    def _vector(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        if self._collection is None:
            self._collection = open_collection(self.persist_dir, self.collection_name)
        return query_collection(
            query,
            model_name=self.model_name,
            top_k=top_k,
            collection=self._collection,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        strategy: str | None = None,
    ) -> list[RetrievedChunk]:
        selected = (strategy or self.strategy).strip().lower()
        if selected not in SUPPORTED_RETRIEVAL_STRATEGIES:
            raise ValueError(f"Unknown retrieval strategy: {selected}")
        candidate_k = max(top_k, self.candidate_k)
        effective_query = rewrite_query(query) if self.enable_query_rewriting else query

        if selected == "bm25":
            return self._lexical_index().search(effective_query, top_k=top_k)
        if selected == "vector":
            return self._vector(effective_query, top_k=top_k)
        if selected == "vector_reranked":
            from cultural_mood_tracker.chat.retrieval_rerank import rerank_chunks

            return rerank_chunks(self._vector(effective_query, top_k=candidate_k), limit=top_k)

        vector = self._vector(effective_query, top_k=candidate_k)
        lexical = self._lexical_index().search(effective_query, top_k=candidate_k)
        return reciprocal_rank_fusion([vector, lexical], top_k=top_k)
