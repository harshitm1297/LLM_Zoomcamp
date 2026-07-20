from __future__ import annotations

import argparse
import json
from pathlib import Path

from cultural_mood_tracker.config import load_settings, make_run_id
from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.evaluation.llm_eval import (
    evaluate_llm_configurations,
    load_llm_golden_set,
)
from cultural_mood_tracker.rag.chroma_ingest import DEFAULT_CHROMA_COLLECTION, DEFAULT_CHROMA_DB_DIR
from cultural_mood_tracker.rag.retriever import ApplicationRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare final-answer prompt configurations.")
    parser.add_argument("--golden-set-path", type=Path, default=Path("data/eval/llm_golden_set.jsonl"))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    project_root = load_project_environment(Path(__file__))
    settings = load_settings()
    args = parse_args()
    golden_path = args.golden_set_path if args.golden_set_path.is_absolute() else project_root / args.golden_set_path
    retriever = ApplicationRetriever(
        strategy=settings.retrieval_strategy,
        persist_dir=project_root / DEFAULT_CHROMA_DB_DIR,
        collection_name=DEFAULT_CHROMA_COLLECTION,
        candidate_k=settings.retrieval_candidate_k,
    )
    report = evaluate_llm_configurations(
        load_llm_golden_set(golden_path),
        retriever=retriever,
        top_k=args.top_k,
    )
    output_path = args.output_path or project_root / "data" / "reports" / "llm_eval" / f"{make_run_id()}.json"
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Best configuration: {report['best_configuration']} ({report['best_score']:.4f})")
    print(f"Full report written to: {output_path}")
    return 0
