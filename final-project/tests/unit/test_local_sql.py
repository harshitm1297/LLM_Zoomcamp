from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from cultural_mood_tracker.chat.sql_client import LocalDuckDBClient
from cultural_mood_tracker.load.local_duckdb import run_local_duckdb_load
from cultural_mood_tracker.pipeline.sample_data import create_sample_processed_run


@unittest.skipUnless(importlib.util.find_spec("duckdb"), "duckdb is not installed")
class LocalSqlTests(unittest.TestCase):
    def test_sample_tables_are_queryable_without_credentials(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        process_run_id = "unit_test_sample"
        create_sample_processed_run(project_root, process_run_id)
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "sample.duckdb"
            run_local_duckdb_load(
                project_root=project_root,
                process_run_id=process_run_id,
                database_path=database,
            )
            client = LocalDuckDBClient(local_path=database)
            try:
                self.assertTrue(client.table_exists("titles"))
                result = client.get_title_ratings("Disclosure Day")
                self.assertEqual(result["results"][0]["title"], "Disclosure Day")
                self.assertGreater(client.get_attention("Disclosure Day")["results"][0]["attention_score"], 0)
                self.assertEqual(len(client.get_cast("Disclosure Day")["results"][0]["cast"]), 2)
                self.assertTrue(client.get_title_theme_summary("science fiction identity"))
                self.assertTrue(client.get_genre_theme_summary("science fiction identity"))
                self.assertTrue(client.get_audience_editorial_summary("science fiction identity"))
                analytics = client.get_title_analytical_summaries("Disclosure Day")
                self.assertTrue(analytics["audience_vs_editorial_summary"])
                self.assertGreater(analytics["attention_vs_reception"]["attention_score"], 0)
                self.assertEqual(client.extract_titles("Compare Disclosure Day and Project Hail Mary")[:2], ["Disclosure Day", "Project Hail Mary"])
            finally:
                client.close()


if __name__ == "__main__":
    unittest.main()
