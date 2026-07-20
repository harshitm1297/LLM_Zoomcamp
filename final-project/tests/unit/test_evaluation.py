from __future__ import annotations

import unittest

from cultural_mood_tracker.evaluation.llm_eval import LLMEvalCase, score_answer
from cultural_mood_tracker.rag.retrieval import RetrievedChunk
from cultural_mood_tracker.rag.retrieval_eval import GoldenQuery, evaluate_retrieval_approaches


class FakeRetriever:
    def retrieve(self, query: str, *, top_k: int, strategy: str | None = None):
        relevant = RetrievedChunk("relevant", "matching context", {}, 0.1)
        distractor = RetrievedChunk("distractor", "other context", {}, 0.2)
        return [relevant, distractor][:top_k] if strategy != "bm25" else [distractor, relevant][:top_k]


class EvaluationTests(unittest.TestCase):
    def test_answer_scoring(self) -> None:
        case = LLMEvalCase("q", "What happened?", "rag", ["missing son"], True)
        scores = score_answer(
            case,
            "He searches New York for his missing son.",
            context="He searches New York for his missing son.",
        )
        self.assertEqual(scores["fact_coverage"], 1.0)
        self.assertEqual(scores["refusal_correctness"], 1.0)

    def test_multiple_retrieval_approaches_select_winner(self) -> None:
        report = evaluate_retrieval_approaches(
            [GoldenQuery("q", "query", ["relevant"])],
            retriever=FakeRetriever(),  # type: ignore[arg-type]
            approaches=("bm25", "vector"),
            k_values=(1, 2, 5),
        )
        self.assertEqual(report["best_approach"], "vector")
        self.assertIn("bm25", report["approaches"])


if __name__ == "__main__":
    unittest.main()
