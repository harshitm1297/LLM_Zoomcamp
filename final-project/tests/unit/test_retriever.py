from __future__ import annotations

import unittest

from cultural_mood_tracker.rag.retrieval import RetrievedChunk
from cultural_mood_tracker.rag.retriever import BM25Index, reciprocal_rank_fusion
from cultural_mood_tracker.rag.query_rewriting import rewrite_query


class RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"chunk_id": "space", "chunk_text": "A scientist wakes alone in space and saves the sun."},
            {"chunk_id": "romance", "chunk_text": "A romantic comedy in a coastal town."},
            {"chunk_id": "crime", "chunk_text": "A detective investigates a conspiracy."},
        ]

    def test_bm25_ranks_matching_document_first(self) -> None:
        results = BM25Index(self.rows).search("scientist space sun", top_k=3)
        self.assertEqual(results[0].chunk_id, "space")

    def test_rank_fusion_rewards_cross_list_evidence(self) -> None:
        a = RetrievedChunk("a", "a", {}, 0.1)
        b = RetrievedChunk("b", "b", {}, 0.2)
        c = RetrievedChunk("c", "c", {}, 0.3)
        fused = reciprocal_rank_fusion([[a, b], [b, c]], top_k=3)
        self.assertEqual(fused[0].chunk_id, "b")

    def test_query_rewriting_expands_search_intent(self) -> None:
        rewritten = rewrite_query("Could you recommend a popular sci-fi film?")
        self.assertIn("science fiction", rewritten)
        self.assertIn("audience attention ratings reviews", rewritten)


if __name__ == "__main__":
    unittest.main()
