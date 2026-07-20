from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    interaction_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    mode TEXT,
    latency_ms REAL,
    error INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    retrieved_chunk_ids TEXT NOT NULL DEFAULT '[]',
    average_similarity REAL,
    model_name TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY,
    interaction_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    score INTEGER NOT NULL CHECK (score IN (-1, 1)),
    comment TEXT,
    FOREIGN KEY (interaction_id) REFERENCES interactions(interaction_id)
);
CREATE INDEX IF NOT EXISTS idx_interactions_created_at ON interactions(created_at);
CREATE INDEX IF NOT EXISTS idx_interactions_mode ON interactions(mode);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at);
"""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class ObservabilityStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        with closing(self.connect()) as connection:
            connection.executescript(SCHEMA)
            connection.commit()

    def record_interaction(
        self,
        *,
        session_id: str,
        query: str,
        answer: str,
        mode: str | None,
        latency_ms: float | None,
        error: bool,
        error_message: str | None = None,
        retrieved_chunk_ids: list[str] | None = None,
        similarities: list[float] | None = None,
        model_name: str | None = None,
    ) -> str:
        interaction_id = str(uuid.uuid4())
        valid_similarities = [float(value) for value in (similarities or [])]
        average_similarity = (
            sum(valid_similarities) / len(valid_similarities) if valid_similarities else None
        )
        with closing(self.connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO interactions (
                        interaction_id, session_id, created_at, query, answer, mode,
                        latency_ms, error, error_message, retrieved_chunk_ids,
                        average_similarity, model_name
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        interaction_id,
                        session_id,
                        _utc_now(),
                        query,
                        answer,
                        mode,
                        latency_ms,
                        int(error),
                        error_message,
                        json.dumps(retrieved_chunk_ids or []),
                        average_similarity,
                        model_name,
                    ),
                )
        return interaction_id

    def record_feedback(self, interaction_id: str, score: int, comment: str | None = None) -> str:
        if score not in {-1, 1}:
            raise ValueError("Feedback score must be -1 or 1")
        feedback_id = str(uuid.uuid4())
        with closing(self.connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO feedback (feedback_id, interaction_id, created_at, score, comment)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(interaction_id) DO UPDATE SET
                        created_at=excluded.created_at,
                        score=excluded.score,
                        comment=excluded.comment
                    """,
                    (feedback_id, interaction_id, _utc_now(), score, comment),
                )
        return feedback_id

    def interactions(self) -> list[dict[str, Any]]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT i.*, f.score AS feedback_score, f.comment AS feedback_comment
                FROM interactions i
                LEFT JOIN feedback f USING (interaction_id)
                ORDER BY i.created_at
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def summary(self) -> dict[str, Any]:
        with closing(self.connect()) as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS requests,
                    SUM(error) AS errors,
                    AVG(latency_ms) AS average_latency_ms,
                    AVG(average_similarity) AS average_similarity
                FROM interactions
                """
            ).fetchone()
            feedback = connection.execute(
                """
                SELECT COUNT(*) AS responses,
                       AVG(CASE WHEN score = 1 THEN 1.0 ELSE 0.0 END) AS positive_rate
                FROM feedback
                """
            ).fetchone()
        return {**dict(row), **dict(feedback)}
