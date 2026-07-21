from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cultural_mood_tracker.pipeline.documents import chunk_documents, prepare_documents_run


class DocumentPipelineTests(unittest.TestCase):
    def test_chunking_uses_overlap_and_preserves_metadata(self) -> None:
        document = {
            "document_id": "movie_1:overview",
            "title_name": "Example",
            "text": "one two three four five six seven",
        }

        chunks = chunk_documents([document], chunk_words=4, overlap_words=1)

        self.assertEqual([row["chunk_text"] for row in chunks], ["one two three four", "four five six seven"])
        self.assertEqual(chunks[0]["chunk_id"], "movie_1:overview:chunk_001")
        self.assertEqual(chunks[0]["title_name"], "Example")

    def test_invalid_chunk_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chunk_documents([], chunk_words=10, overlap_words=10)

    def test_prepare_run_writes_only_rag_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            documents = [
                {
                    "document_id": "movie_1:overview",
                    "title_name": "Example",
                    "content_type": "movie",
                    "text": "A compact source passage.",
                }
            ]
            with patch.dict(os.environ, {"LOCAL_DATA_ROOT": "data"}, clear=False):
                run_id, chunk_path = prepare_documents_run(
                    root, documents, run_id="test_run", source="sample"
                )

            self.assertEqual(run_id, "test_run")
            self.assertTrue(chunk_path.is_file())
            run_dir = root / "data" / "processed" / run_id
            self.assertEqual(
                {path.name for path in run_dir.iterdir()},
                {"documents.jsonl", "document_chunks.jsonl", "run_manifest.json"},
            )
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["documents"], 1)
            self.assertEqual(manifest["chunks"], 1)
            self.assertEqual(manifest["source"], "sample")


if __name__ == "__main__":
    unittest.main()
