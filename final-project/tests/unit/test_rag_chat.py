from __future__ import annotations

import unittest
from unittest.mock import patch

from cultural_mood_tracker.chat import ChatOrchestrator
from cultural_mood_tracker.rag.retrieval import RetrievedChunk


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def retrieve(self, query: str, *, top_k: int, strategy: str):
        self.calls.append({"query": query, "top_k": top_k, "strategy": strategy})
        return [
            RetrievedChunk(
                chunk_id="movie:overview:1",
                chunk_text="A scientist wakes alone in space and must save humanity.",
                metadata={"title_name": "Project Hail Mary", "source_name": "sample"},
                distance=0.09,
            )
        ]


class RagChatTests(unittest.TestCase):
    def test_every_question_uses_vector_rag_without_sql(self) -> None:
        retriever = FakeRetriever()
        chatbot = ChatOrchestrator(retriever=retriever, top_k=4)

        with patch(
            "cultural_mood_tracker.chat.orchestrator.generate_answer",
            return_value="A grounded answer [1].",
        ):
            result = chatbot.answer("What happens in Project Hail Mary?")

        self.assertEqual(result.mode, "rag")
        self.assertFalse(result.used_sql)
        self.assertEqual(result.answer, "A grounded answer [1].")
        self.assertEqual(result.retrieved_chunk_ids, ["movie:overview:1"])
        self.assertEqual(retriever.calls[0]["strategy"], "vector")
        self.assertEqual(retriever.calls[0]["top_k"], 4)

    def test_empty_question_is_rejected(self) -> None:
        chatbot = ChatOrchestrator(retriever=FakeRetriever())
        with self.assertRaises(ValueError):
            chatbot.answer("  ")


if __name__ == "__main__":
    unittest.main()
