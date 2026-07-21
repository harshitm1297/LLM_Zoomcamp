from __future__ import annotations

import os
from pathlib import Path

from cultural_mood_tracker.pipeline import bootstrap_application
from cultural_mood_tracker.rag import DEFAULT_CHROMA_DB_DIR


def cloud_data_is_ready(project_root: Path) -> bool:
    """Return whether the persistent vector collection has been initialized."""
    chroma_path = Path(DEFAULT_CHROMA_DB_DIR)
    if not chroma_path.is_absolute():
        chroma_path = project_root / chroma_path
    return (chroma_path / "chroma.sqlite3").is_file()


def ensure_cloud_data(project_root: Path) -> bool:
    if cloud_data_is_ready(project_root):
        return False
    bootstrap_application(project_root=project_root, sample=True)
    return True


def streamlit_command(project_root: Path, port: str) -> list[str]:
    return [
        "streamlit",
        "run",
        str(project_root / "app.py"),
        "--server.address=0.0.0.0",
        f"--server.port={port}",
        "--server.headless=true",
    ]


def main() -> int:
    project_root = Path(__file__).resolve().parents[3]
    initialized = ensure_cloud_data(project_root)
    print(
        "Initialized the sample vector index." if initialized else "Using the existing vector index.",
        flush=True,
    )
    command = streamlit_command(project_root, os.getenv("PORT", "8501"))
    os.execvp(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
