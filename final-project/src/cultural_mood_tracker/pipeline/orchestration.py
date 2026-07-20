from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cultural_mood_tracker.config import load_settings, make_run_id
from cultural_mood_tracker.load import run_local_duckdb_load
from cultural_mood_tracker.transform.common import write_json


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_pipeline(
    *,
    project_root: Path,
    extract_fn,
    extract_args,
    transform_fn,
    source_run_id: str | None = None,
) -> dict[str, Any]:
    """Download, transform, and publish all data to a local DuckDB database."""
    settings = load_settings()
    paths = settings.build_paths(project_root)
    paths.ensure()
    pipeline_run_id = make_run_id()
    manifest_path = paths.reports_root / pipeline_run_id / "pipeline_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "pipeline_run_id": pipeline_run_id,
        "started_at_utc": _utc_now(),
        "storage": "local",
        "status": "running",
        "steps": [],
    }
    write_json(manifest_path, manifest)

    def begin_step(step: str, **details: Any) -> None:
        manifest["steps"].append(
            {"step": step, "status": "running", "started_at_utc": _utc_now(), **details}
        )
        write_json(manifest_path, manifest)

    def complete_step(**details: Any) -> None:
        manifest["steps"][-1].update(
            {"status": "completed", "finished_at_utc": _utc_now(), **details}
        )
        write_json(manifest_path, manifest)

    try:
        resolved_source_run_id = source_run_id
        if not resolved_source_run_id:
            begin_step("download_sources")
            resolved_source_run_id = extract_fn(project_root, extract_args)
            complete_step(source_run_id=resolved_source_run_id)

        begin_step("transform_canonical", source_run_id=resolved_source_run_id)
        process_run_id = transform_fn(project_root, resolved_source_run_id, None)
        complete_step(process_run_id=process_run_id)

        begin_step("load_local_duckdb", process_run_id=process_run_id)
        load_manifest = run_local_duckdb_load(
            project_root=project_root, process_run_id=process_run_id
        )
        complete_step(load_manifest_path=load_manifest["manifest_path"])

        manifest.update(
            {
                "status": "completed",
                "finished_at_utc": _utc_now(),
                "source_run_id": resolved_source_run_id,
                "process_run_id": process_run_id,
                "local_duckdb_manifest_path": load_manifest["manifest_path"],
            }
        )
        write_json(manifest_path, manifest)
        return manifest
    except Exception as exc:  # noqa: BLE001
        if manifest["steps"] and manifest["steps"][-1]["status"] == "running":
            manifest["steps"][-1].update(
                {"status": "failed", "finished_at_utc": _utc_now(), "error": str(exc)}
            )
        manifest.update({"status": "failed", "finished_at_utc": _utc_now(), "error": str(exc)})
        write_json(manifest_path, manifest)
        raise
