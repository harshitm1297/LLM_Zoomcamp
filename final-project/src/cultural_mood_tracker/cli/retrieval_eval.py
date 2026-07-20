from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from cultural_mood_tracker.config import load_settings, make_run_id
from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.rag.chroma_ingest import DEFAULT_CHROMA_COLLECTION, DEFAULT_CHROMA_DB_DIR
from cultural_mood_tracker.rag.embeddings import DEFAULT_EMBEDDING_MODEL
from cultural_mood_tracker.rag.retrieval_eval import (
    DEFAULT_GOLDEN_SET_PATH,
    DEFAULT_K_VALUES,
    evaluate_retrieval_approaches,
    load_golden_set,
)
from cultural_mood_tracker.rag.retriever import ApplicationRetriever, SUPPORTED_RETRIEVAL_STRATEGIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score retrieval quality against a hand-labeled golden set of (query -> relevant chunk_ids)."
    )
    parser.add_argument(
        "--approaches",
        nargs="+",
        choices=sorted(SUPPORTED_RETRIEVAL_STRATEGIES),
        default=["bm25", "vector", "vector_reranked", "hybrid"],
        help="Retrieval approaches to compare.",
    )
    parser.add_argument(
        "--select-by",
        choices=["mrr", "recall_at_5", "precision_at_5"],
        default="mrr",
        help="Metric used to select the production winner.",
    )
    parser.add_argument(
        "--golden-set-path",
        default=Path(DEFAULT_GOLDEN_SET_PATH),
        type=Path,
        help=f"Path to the golden set JSONL file. Defaults to {DEFAULT_GOLDEN_SET_PATH}.",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=list(DEFAULT_K_VALUES),
        help=f"k values to compute recall@k/precision@k for. Defaults to {list(DEFAULT_K_VALUES)}.",
    )
    parser.add_argument(
        "--persist-dir",
        default=Path(DEFAULT_CHROMA_DB_DIR),
        type=Path,
        help=f"Directory of the persistent ChromaDB database. Defaults to ./{DEFAULT_CHROMA_DB_DIR}.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_CHROMA_COLLECTION,
        help=f"ChromaDB collection name. Defaults to {DEFAULT_CHROMA_COLLECTION}.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"SentenceTransformer model name. Must match the model used at ingest time. Defaults to {DEFAULT_EMBEDDING_MODEL}.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to write the full report as JSON. Defaults to data/reports/retrieval_eval/<run_id>.json.",
    )
    return parser.parse_args()


def _print_report(report: dict) -> None:
    if "approaches" in report:
        print(f"Best approach: {report['best_approach']} ({report['selection_metric']}={report['best_score']:.4f})")
        print("\napproach".ljust(24) + "MRR".ljust(12) + "Recall@5".ljust(12) + "Latency ms")
        for name, approach_report in report["approaches"].items():
            aggregate = approach_report["aggregate"]
            recall_values = aggregate["recall_at_k"]
            recall_5 = recall_values.get(5, recall_values.get("5", 0.0))
            print(
                name.ljust(24)
                + f"{aggregate['mrr']:.4f}".ljust(12)
                + f"{recall_5:.4f}".ljust(12)
                + f"{aggregate['mean_latency_ms']:.2f}"
            )
        return
    aggregate = report["aggregate"]
    k_values = report["k_values"]

    print(f"Golden set queries: {report['num_queries']}")
    print(f"Collection: {report['collection_name']} @ {report['persist_dir']}")
    print(f"Model: {report['model_name']}\n")

    print(f"MRR: {aggregate['mrr']:.4f}\n")
    header = "k".ljust(6) + "recall@k".ljust(12) + "precision@k"
    print(header)
    for k in k_values:
        recall = aggregate["recall_at_k"][k]
        precision = aggregate["precision_at_k"][k]
        print(f"{str(k).ljust(6)}{f'{recall:.4f}'.ljust(12)}{precision:.4f}")

    if report["missing_relevant_chunk_ids"]:
        print(
            f"\nWARNING: {len(report['missing_relevant_chunk_ids'])} relevant chunk_id(s) from the "
            "golden set were not found in the collection at all (corpus drift, not a retrieval failure):"
        )
        for chunk_id in report["missing_relevant_chunk_ids"]:
            print(f"  - {chunk_id}")

    print("\nWeakest queries (lowest reciprocal rank):")
    worst = sorted(report["per_query"], key=lambda r: r["reciprocal_rank"])[:5]
    for row in worst:
        print(f"  [{row['reciprocal_rank']:.3f}] {row['query_id']}: {row['query']!r}")


def main() -> int:
    project_root = load_project_environment(Path(__file__))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    golden_set_path = args.golden_set_path
    if not golden_set_path.is_absolute():
        golden_set_path = project_root / golden_set_path

    persist_dir = args.persist_dir
    if not persist_dir.is_absolute():
        persist_dir = project_root / persist_dir

    golden_set = load_golden_set(golden_set_path)
    retriever = ApplicationRetriever(
        strategy=args.approaches[0],
        persist_dir=persist_dir,
        collection_name=args.collection_name,
        model_name=args.model_name,
        candidate_k=max(args.k),
    )
    report = evaluate_retrieval_approaches(
        golden_set,
        retriever=retriever,
        approaches=tuple(args.approaches),
        k_values=tuple(sorted(set(args.k))),
        select_by=args.select_by,
    )

    _print_report(report)

    settings = load_settings()
    paths = settings.build_paths(project_root)
    output_path = args.output_path
    if output_path is None:
        output_path = paths.reports_root / "retrieval_eval" / f"{make_run_id()}.json"
    elif not output_path.is_absolute():
        output_path = project_root / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull report written to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
