from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .retriever import ApplicationRetriever, SUPPORTED_RETRIEVAL_STRATEGIES

DEFAULT_GOLDEN_SET_PATH = "data/eval/retrieval_golden_set.jsonl"
DEFAULT_K_VALUES = (1, 3, 5, 10)


@dataclass(frozen=True)
class GoldenQuery:
    query_id: str
    query: str
    relevant_chunk_ids: list[str]
    difficulty: str | None = None
    notes: str | None = None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def load_golden_set(path: Path) -> list[GoldenQuery]:
    """Load a hand-labeled (query -> relevant chunk_ids) evaluation set.

    This is a curated ground-truth file, not pipeline output: it records what a human
    (or someone reading the corpus closely) judged to be the relevant chunk(s) for a
    given natural-language query. See data/eval/retrieval_golden_set.jsonl for the format.
    """
    if not path.exists():
        raise RuntimeError(f"Golden set file not found: {path}")

    queries: list[GoldenQuery] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(_load_jsonl(path), start=1):
        query_id = row.get("query_id")
        query = row.get("query")
        relevant_chunk_ids = row.get("relevant_chunk_ids")

        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"Golden set row {index} is missing a non-empty string query_id")
        if query_id in seen_ids:
            raise ValueError(f"Duplicate query_id in golden set: {query_id}")
        seen_ids.add(query_id)
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Golden set row {index} ({query_id}) is missing non-empty query text")
        if not isinstance(relevant_chunk_ids, list) or not relevant_chunk_ids:
            raise ValueError(f"Golden set row {index} ({query_id}) needs a non-empty relevant_chunk_ids list")

        queries.append(
            GoldenQuery(
                query_id=query_id,
                query=query,
                relevant_chunk_ids=[str(cid) for cid in relevant_chunk_ids],
                difficulty=row.get("difficulty"),
                notes=row.get("notes"),
            )
        )

    return queries


def _recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top = set(retrieved_ids[:k])
    return len(top & relevant_ids) / len(relevant_ids)


def _precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    return len(set(top) & relevant_ids) / len(top)


def _reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retriever(
    golden_set: list[GoldenQuery],
    *,
    retriever: ApplicationRetriever,
    strategy: str,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    """Evaluate any configured retrieval strategy over the same golden set."""
    if not golden_set:
        raise ValueError("golden_set must not be empty")
    max_k = max(k_values)
    per_query: list[dict[str, Any]] = []
    latencies: list[float] = []
    for item in golden_set:
        started = time.perf_counter()
        retrieved = retriever.retrieve(item.query, top_k=max_k, strategy=strategy)
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        retrieved_ids = [chunk.chunk_id for chunk in retrieved]
        relevant_ids = set(item.relevant_chunk_ids)
        per_query.append(
            {
                "query_id": item.query_id,
                "query": item.query,
                "difficulty": item.difficulty,
                "relevant_chunk_ids": item.relevant_chunk_ids,
                "retrieved_chunk_ids": retrieved_ids,
                "reciprocal_rank": _reciprocal_rank(retrieved_ids, relevant_ids),
                "recall_at_k": {k: _recall_at_k(retrieved_ids, relevant_ids, k) for k in k_values},
                "precision_at_k": {k: _precision_at_k(retrieved_ids, relevant_ids, k) for k in k_values},
                "latency_ms": latency_ms,
            }
        )

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    aggregate = {
        "mrr": mean([row["reciprocal_rank"] for row in per_query]),
        "recall_at_k": {
            k: mean([row["recall_at_k"][k] for row in per_query]) for k in k_values
        },
        "precision_at_k": {
            k: mean([row["precision_at_k"][k] for row in per_query]) for k in k_values
        },
        "mean_latency_ms": mean(latencies),
    }
    return {
        "strategy": strategy,
        "num_queries": len(per_query),
        "k_values": list(k_values),
        "aggregate": aggregate,
        "per_query": per_query,
    }


def evaluate_retrieval_approaches(
    golden_set: list[GoldenQuery],
    *,
    retriever: ApplicationRetriever,
    approaches: tuple[str, ...] = ("bm25", "vector", "vector_reranked", "hybrid"),
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    select_by: str = "mrr",
) -> dict[str, Any]:
    invalid = sorted(set(approaches) - SUPPORTED_RETRIEVAL_STRATEGIES)
    if invalid:
        raise ValueError(f"Unsupported retrieval approaches: {', '.join(invalid)}")
    if select_by not in {"mrr", "recall_at_5", "precision_at_5"}:
        raise ValueError("select_by must be mrr, recall_at_5, or precision_at_5")

    reports = {
        approach: evaluate_retriever(
            golden_set,
            retriever=retriever,
            strategy=approach,
            k_values=k_values,
        )
        for approach in approaches
    }

    def selection_value(report: dict[str, Any]) -> float:
        aggregate = report["aggregate"]
        if select_by == "mrr":
            return float(aggregate["mrr"])
        metric, raw_k = select_by.rsplit("_at_", 1)
        values = aggregate[f"{metric}_at_k"]
        return float(values.get(int(raw_k), values.get(str(int(raw_k)), 0.0)))

    best_approach = max(
        approaches,
        key=lambda approach: (selection_value(reports[approach]), -approaches.index(approach)),
    )
    return {
        "approaches": reports,
        "selection_metric": select_by,
        "best_approach": best_approach,
        "best_score": selection_value(reports[best_approach]),
    }
