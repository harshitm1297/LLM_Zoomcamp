from __future__ import annotations

import json
import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.pipeline import bootstrap_application


def main() -> int:
    """Optional maintainer workflow: refresh TMDB and rebuild the vector index."""
    project_root = load_project_environment(Path(__file__))
    manifest = bootstrap_application(project_root=project_root, sample=False)
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
