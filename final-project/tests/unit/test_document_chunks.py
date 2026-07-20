from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cultural_mood_tracker.rag.document_chunks import resolve_local_document_chunks_path


class DocumentChunkResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_run(self, run_id: str) -> Path:
        path = self.root / "data" / "processed" / run_id / "document_chunks.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text('{"chunk_id":"x","chunk_text":"text"}\n', encoding="utf-8")
        return path

    def test_explicit_path_wins(self) -> None:
        explicit = self.root / "explicit.jsonl"
        explicit.write_text("{}\n", encoding="utf-8")
        with patch.dict(os.environ, {"DOCUMENT_CHUNKS_PATH": "", "PROCESS_RUN_ID": ""}, clear=False):
            self.assertEqual(
                resolve_local_document_chunks_path(self.root, input_path=explicit),
                explicit,
            )

    def test_process_run_id_is_resolved(self) -> None:
        expected = self._write_run("run_a")
        with patch.dict(os.environ, {"DOCUMENT_CHUNKS_PATH": "", "PROCESS_RUN_ID": ""}, clear=False):
            self.assertEqual(
                resolve_local_document_chunks_path(self.root, process_run_id="run_a"),
                expected,
            )

    def test_latest_valid_run_is_selected(self) -> None:
        self._write_run("20260101T000000Z")
        expected = self._write_run("20260201T000000Z")
        (self.root / "data" / "processed" / "20260301T000000Z").mkdir(parents=True)
        with patch.dict(os.environ, {"DOCUMENT_CHUNKS_PATH": "", "PROCESS_RUN_ID": ""}, clear=False):
            self.assertEqual(resolve_local_document_chunks_path(self.root), expected)


if __name__ == "__main__":
    unittest.main()
