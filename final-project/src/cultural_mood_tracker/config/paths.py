from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_root: Path
    processed_root: Path
    reports_root: Path
    logs_root: Path

    def ensure(self) -> None:
        for path in (
            self.data_root,
            self.processed_root,
            self.reports_root,
            self.logs_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def run_root(self, layer: str, run_id: str) -> Path:
        base = {
            "processed": self.processed_root,
            "reports": self.reports_root,
            "logs": self.logs_root,
        }[layer]
        return base / run_id

def build_project_paths(project_root: Path, data_root: str | Path = "data") -> ProjectPaths:
    resolved_project_root = project_root.resolve()
    resolved_data_root = (resolved_project_root / Path(data_root)).resolve()
    return ProjectPaths(
        project_root=resolved_project_root,
        data_root=resolved_data_root,
        processed_root=resolved_data_root / "processed",
        reports_root=resolved_data_root / "reports",
        logs_root=resolved_data_root / "logs",
    )


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
