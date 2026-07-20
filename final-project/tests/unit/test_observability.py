from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cultural_mood_tracker.observability import ObservabilityStore


class ObservabilityTests(unittest.TestCase):
    def test_interaction_and_feedback_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ObservabilityStore(Path(directory) / "observability.db")
            interaction_id = store.record_interaction(
                session_id="session",
                query="question",
                answer="answer",
                mode="rag",
                latency_ms=12.5,
                error=False,
                retrieved_chunk_ids=["chunk"],
                similarities=[0.75],
                model_name="test-model",
            )
            store.record_feedback(interaction_id, 1)
            store.record_feedback(interaction_id, -1, "Updated")
            rows = store.interactions()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["feedback_score"], -1)
            self.assertEqual(store.summary()["responses"], 1)


if __name__ == "__main__":
    unittest.main()
