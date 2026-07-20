from __future__ import annotations

import os
from pathlib import Path

from cultural_mood_tracker.config import load_settings
from cultural_mood_tracker.pipeline import bootstrap_application
from cultural_mood_tracker.rag import DEFAULT_CHROMA_DB_DIR


def _resolve(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def cloud_data_is_ready(project_root: Path) -> bool:
    """Return whether the local SQL and vector stores have been initialized."""
    settings = load_settings()
    duckdb_path = _resolve(project_root, settings.local_duckdb_path)
    chroma_path = _resolve(project_root, Path(DEFAULT_CHROMA_DB_DIR))
    return duckdb_path.is_file() and (chroma_path / "chroma.sqlite3").is_file()


def ensure_cloud_data(project_root: Path) -> bool:
    """Initialize the bundled sample corpus only when a cloud disk is empty."""
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
    if initialized:
        print("Initialized the persistent cloud disk with the sample corpus.", flush=True)
    else:
        print("Using the existing persistent cloud data.", flush=True)

    command = streamlit_command(project_root, os.getenv("PORT", "8501"))
    os.execvp(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
