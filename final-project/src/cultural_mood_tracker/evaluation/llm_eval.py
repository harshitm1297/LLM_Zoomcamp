from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from cultural_mood_tracker.rag.llm import DEFAULT_MODEL, generate_answer
from cultural_mood_tracker.rag.prompting import build_prompt, system_instruction_for_variant
from cultural_mood_tracker.rag.retriever import ApplicationRetriever


REFUSAL_MARKERS = (
    "not enough information",
    "insufficient context",
    "cannot answer",
    "can't answer",
    "not provided in the context",
    "no relevant context",
)
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "with"}


@dataclass(frozen=True)
class LLMEvalCase:
    query_id: str
    question: str
    category: str
    expected_facts: list[str]
    answerable: bool


DEFAULT_CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "baseline": {
        "prompt_variant": "baseline",
        "model_name": DEFAULT_MODEL,
        "max_context_chars": 3000,
        "temperature": 0.2,
    },
    "strict": {
        "prompt_variant": "strict",
        "model_name": DEFAULT_MODEL,
        "max_context_chars": 6000,
        "temperature": 0.0,
    },
}


def load_llm_golden_set(path: Path) -> list[LLMEvalCase]:
    cases: list[LLMEvalCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                cases.append(
                    LLMEvalCase(
                        query_id=str(row["query_id"]),
                        question=str(row["question"]),
                        category=str(row.get("category") or "rag"),
                        expected_facts=[str(value) for value in row.get("expected_facts", [])],
                        answerable=bool(row.get("answerable", True)),
                    )
                )
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Invalid LLM golden-set row {line_number}: {row}") from exc
    if not cases:
        raise ValueError("LLM golden set must not be empty")
    return cases


def _tokens(text: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(text.casefold()) if token not in STOPWORDS}


def score_answer(
    case: LLMEvalCase,
    answer: str,
    *,
    context: str,
) -> dict[str, float | bool]:
    normalized_answer = " ".join(answer.casefold().split())
    fact_hits = [
        " ".join(fact.casefold().split()) in normalized_answer
        for fact in case.expected_facts
    ]
    fact_coverage = sum(fact_hits) / len(fact_hits) if fact_hits else 1.0
    refused = any(marker in normalized_answer for marker in REFUSAL_MARKERS)
    refusal_correctness = float(refused if not case.answerable else not refused)
    answer_tokens = _tokens(answer)
    context_tokens = _tokens(context)
    grounding_overlap = (
        len(answer_tokens & context_tokens) / len(answer_tokens) if answer_tokens else 0.0
    )
    composite = 0.5 * fact_coverage + 0.3 * grounding_overlap + 0.2 * refusal_correctness
    return {
        "fact_coverage": fact_coverage,
        "grounding_overlap": grounding_overlap,
        "refusal_correctness": refusal_correctness,
        "refused": refused,
        "composite_score": composite,
    }


def evaluate_llm_configurations(
    cases: list[LLMEvalCase],
    *,
    retriever: ApplicationRetriever,
    configurations: dict[str, dict[str, Any]] | None = None,
    top_k: int = 5,
    answer_generator: Callable[..., str] = generate_answer,
) -> dict[str, Any]:
    configs = configurations or DEFAULT_CONFIGURATIONS
    results: dict[str, Any] = {}
    for config_name, config in configs.items():
        per_case: list[dict[str, Any]] = []
        for case in cases:
            chunks = retriever.retrieve(case.question, top_k=top_k)
            prompt = build_prompt(
                case.question,
                chunks,
                max_context_chars=int(config["max_context_chars"]),
                system_instruction=system_instruction_for_variant(str(config["prompt_variant"])),
            )
            started = time.perf_counter()
            answer = answer_generator(
                prompt,
                model_name=str(config["model_name"]),
                temperature=float(config["temperature"]),
            )
            latency_ms = (time.perf_counter() - started) * 1000
            context = "\n".join(chunk.chunk_text for chunk in chunks)
            scores = score_answer(case, answer, context=context)
            per_case.append(
                {
                    "query_id": case.query_id,
                    "question": case.question,
                    "category": case.category,
                    "answer": answer,
                    "retrieved_chunk_ids": [chunk.chunk_id for chunk in chunks],
                    "latency_ms": latency_ms,
                    **scores,
                }
            )

        def mean(key: str) -> float:
            return sum(float(row[key]) for row in per_case) / len(per_case)

        results[config_name] = {
            "configuration": config,
            "aggregate": {
                "fact_coverage": mean("fact_coverage"),
                "grounding_overlap": mean("grounding_overlap"),
                "refusal_correctness": mean("refusal_correctness"),
                "composite_score": mean("composite_score"),
                "mean_latency_ms": mean("latency_ms"),
            },
            "per_case": per_case,
        }

    best_configuration = max(
        results,
        key=lambda name: (results[name]["aggregate"]["composite_score"], name),
    )
    return {
        "configurations": results,
        "selection_metric": "composite_score",
        "best_configuration": best_configuration,
        "best_score": results[best_configuration]["aggregate"]["composite_score"],
    }
