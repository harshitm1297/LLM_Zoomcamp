from __future__ import annotations

import argparse
import json
from pathlib import Path

from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.pipeline.bootstrap import bootstrap_application


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the local data stores for the RAG application.")
    parser.add_argument("--sample", action="store_true", help="Generate a small local sample corpus.")
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--process-run-id", default=None)
    parser.add_argument("--skip-vector-index", action="store_true")
    return parser.parse_args()


def main() -> int:
    project_root = load_project_environment(Path(__file__))
    args = parse_args()
    manifest = bootstrap_application(
        project_root=project_root,
        sample=args.sample,
        source_run_id=args.source_run_id,
        process_run_id=args.process_run_id,
        skip_vector_index=args.skip_vector_index,
    )
    print(json.dumps(manifest, indent=2, default=str))
    return 0
