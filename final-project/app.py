"""Streamlit chat frontend for the Cultural Mood Tracker orchestrator.

This file is a pure UI layer: it imports and calls the existing chatbot backend
(`cultural_mood_tracker.chat.orchestrator.ChatOrchestrator`) and never reimplements
routing, retrieval, SQL, or prompt/generation logic. `scripts/chat.py` (the terminal
entry point) is untouched and keeps working exactly as before -- this is an
additional frontend on top of the same backend, not a replacement.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cultural_mood_tracker.chat.orchestrator import ChatOrchestrator
from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.rag.chroma_ingest import DEFAULT_CHROMA_COLLECTION, DEFAULT_CHROMA_DB_DIR
from cultural_mood_tracker.rag.embeddings import DEFAULT_EMBEDDING_MODEL
from cultural_mood_tracker.rag.llm import DEFAULT_MODEL
from cultural_mood_tracker.rag.prompting import DEFAULT_MAX_CONTEXT_CHARS
from cultural_mood_tracker.config import load_settings
from cultural_mood_tracker.observability import ObservabilityStore


# Same mode -> color mapping used in the terminal (scripts/chat.py)'s rich output, so the two
# frontends stay visually consistent for anyone using both.
MODE_INFO: dict[str, dict[str, str]] = {
    "fast_sql": {"label": "FAST SQL", "color": "#5fc9d4", "description": "Answered directly from compact local DuckDB analytics; skips embedding, ChromaDB, and LLM."},
    "sql": {"label": "SQL", "color": "#5fc9d4", "description": "Answered from structured local DuckDB facts."},
    "rag": {"label": "RAG", "color": "#6fcf7d", "description": "Answered from a small set of retrieved ChromaDB review/summary chunks."},
    "hybrid": {"label": "HYBRID", "color": "#d88ce8", "description": "Answered from SQL metrics first, then compact ChromaDB evidence for interpretation."},
    "recommendation": {"label": "RECOMMENDATION", "color": "#f2c14e", "description": "SQL-ranked recommendations enriched with one exact-title review excerpt per candidate."},
}


# ---------------------------------------------------------------------------
# Page setup + theme
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Cultural Mood Tracker", page_icon="🎬", layout="centered")

CINEMA_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Inter:wght@400;500;600&display=swap');

.stApp {
    background: radial-gradient(circle at top, #181b22 0%, #0a0b0f 60%);
    color: #eae6df;
}
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Marquee-style header */
.marquee-header {
    text-align: center;
    padding: 0.4rem 0 1.1rem 0;
    border-bottom: 1px solid rgba(242, 193, 78, 0.25);
    margin-bottom: 1.4rem;
}
.marquee-title {
    font-family: 'Playfair Display', serif;
    font-weight: 800;
    font-size: 2.5rem;
    letter-spacing: 0.03em;
    background: linear-gradient(135deg, #f2c14e 20%, #e3903a 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.marquee-subtitle {
    color: #9a9eab;
    font-size: 0.95rem;
    margin-top: 0.4rem;
    letter-spacing: 0.01em;
}

/* Metadata row under an assistant answer */
.meta-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 0.5rem 0 0.2rem 0;
    flex-wrap: wrap;
}
.mode-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.18rem 0.7rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border: 1px solid currentColor;
    background: rgba(255, 255, 255, 0.03);
}
.mode-dot {
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 50%;
    background: currentColor;
    display: inline-block;
}
.elapsed-pill {
    font-size: 0.72rem;
    color: #9a9eab;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.12);
}
.sql-flag {
    font-size: 0.72rem;
    color: #9a9eab;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0e13;
    border-right: 1px solid rgba(242, 193, 78, 0.12);
}
.sidebar-title {
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    font-size: 1.3rem;
    color: #f2c14e;
    margin-bottom: 0.2rem;
}
.sidebar-caption {
    color: #9a9eab;
    font-size: 0.85rem;
    line-height: 1.4rem;
}
.legend-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0.3rem 0;
    font-size: 0.85rem;
}
.detail-caption {
    color: #9a9eab;
    font-size: 0.82rem;
    line-height: 1.35rem;
}
</style>
"""

st.markdown(CINEMA_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="marquee-header">
        <p class="marquee-title">Cultural Mood Tracker</p>
        <p class="marquee-subtitle">Ask about movies &amp; TV shows &mdash; ratings, reviews, trends, and recommendations</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Backend resources
#
# The orchestrator is stored in Streamlit session state below so its mutable SQL history and
# connection remain isolated per visitor. Immutable embedding-model and LLM clients retain
# their module-level caches. The observability store is safe to share because it opens a
# short-lived SQLite connection for each operation.
# ---------------------------------------------------------------------------
def get_orchestrator() -> ChatOrchestrator:
    project_root = load_project_environment(Path(__file__))
    settings = load_settings()
    persist_dir = Path(DEFAULT_CHROMA_DB_DIR)
    if not persist_dir.is_absolute():
        persist_dir = project_root / persist_dir
    orchestrator = ChatOrchestrator(
        persist_dir=persist_dir,
        collection_name=DEFAULT_CHROMA_COLLECTION,
        embedding_model_name=DEFAULT_EMBEDDING_MODEL,
        llm_model_name=DEFAULT_MODEL,
        top_k=settings.retrieval_top_k,
        retrieval_strategy=settings.retrieval_strategy,
        candidate_k=settings.retrieval_candidate_k,
        max_context_chars=DEFAULT_MAX_CONTEXT_CHARS,
    )
    orchestrator.healthcheck()
    return orchestrator


@st.cache_resource(show_spinner=False)
def get_observability_store() -> ObservabilityStore:
    project_root = load_project_environment(Path(__file__))
    settings = load_settings()
    path = settings.observability_db_path
    if not path.is_absolute():
        path = project_root / path
    return ObservabilityStore(path)


def _friendly_error(exc: Exception) -> str:
    """Translate a raw backend exception into an actionable message for the chat window."""
    message = str(exc)
    lowered = message.casefold()
    if "groq_api_key" in lowered or "groq generation failed" in lowered or "groq returned" in lowered:
        return f"**The Groq LLM call failed.** Check that `GROQ_API_KEY` is set correctly in `.env`.\n\nDetails: {message}"
    if "duckdb" in lowered or "structured database" in lowered:
        return f"**The local DuckDB query failed.** Run `python scripts/bootstrap.py --sample` to rebuild the local data stores.\n\nDetails: {message}"
    if "chromadb" in lowered or "collection" in lowered or "persist" in lowered:
        return f"**The local ChromaDB retrieval step failed.** Make sure `chroma_db/` has been populated (see README step 5).\n\nDetails: {message}"
    return f"**Something went wrong while generating this answer.**\n\nDetails: {message}"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages: list[dict[str, Any]] = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


def _mode_badge_html(mode: str | None) -> str:
    info = MODE_INFO.get(mode or "", {"label": (mode or "unknown").upper(), "color": "#9a9eab"})
    return (
        f'<span class="mode-badge" style="color:{info["color"]};">'
        f'<span class="mode-dot"></span>{info["label"]}</span>'
    )


def _render_meta(entry: dict[str, Any]) -> None:
    badge = _mode_badge_html(entry.get("mode"))
    elapsed = entry.get("elapsed")
    elapsed_html = f'<span class="elapsed-pill">{elapsed:.2f}s</span>' if elapsed is not None else ""
    sql_flag_html = '<span class="sql-flag">uses SQL</span>' if entry.get("used_sql") else ""
    st.markdown(f'<div class="meta-row">{badge}{elapsed_html}{sql_flag_html}</div>', unsafe_allow_html=True)

    sql_results = entry.get("sql_results") or {}
    if sql_results:
        _render_structured_data(entry.get("mode"), sql_results)

    sql_queries = entry.get("sql_queries") or []
    if sql_queries:
        with st.expander(f"SQL queries used ({len(sql_queries)})"):
            for index, sql in enumerate(sql_queries, start=1):
                st.caption(f"Query {index}")
                st.code(sql, language="sql")

    evidence = entry.get("evidence") or []
    if evidence:
        label = "Recommendation review excerpts" if entry.get("mode") == "recommendation" else "Retrieved evidence"
        with st.expander(f"{label} ({len(evidence)})"):
            for index, item in enumerate(evidence, start=1):
                title = item.get("title") or "Unknown title"
                source = item.get("source") or "unknown source"
                document_type = item.get("document_type") or "text"
                similarity = item.get("similarity")
                lookup = item.get("lookup")
                similarity_str = (
                    lookup
                    if lookup
                    else f"similarity {similarity:.3f}" if isinstance(similarity, (int, float))
                    else "exact-title lookup"
                )
                st.markdown(f"**{index}. {title}** &mdash; {source} / {document_type} ({similarity_str})")
                st.caption(item.get("chunk_id", ""))
                snippet = item.get("snippet")
                if snippet:
                    st.markdown(f'<p class="detail-caption">{snippet}</p>', unsafe_allow_html=True)


def _render_feedback(entry: dict[str, Any]) -> None:
    interaction_id = entry.get("interaction_id")
    if not interaction_id or entry.get("error"):
        return
    selection = st.feedback("thumbs", key=f"feedback_{interaction_id}")
    if selection is not None:
        get_observability_store().record_feedback(
            interaction_id,
            1 if selection == 1 else -1,
        )
        st.caption("Feedback saved. Thank you.")


def _render_structured_data(mode: str | None, sql_results: dict[str, Any]) -> None:
    if mode == "recommendation":
        candidates = sql_results.get("recommendation_candidates") or []
        genres = sql_results.get("genre_theme_summary") or []
        if candidates:
            with st.expander(f"SQL-ranked recommendation candidates ({len(candidates)})"):
                for index, candidate in enumerate(candidates, start=1):
                    title = candidate.get("title") or "Unknown title"
                    themes = _join_list(candidate.get("dominant_themes") or candidate.get("audience_themes") or candidate.get("editorial_themes"), 4)
                    genres_text = _join_list(candidate.get("genres"), 3)
                    rating = candidate.get("avg_rating")
                    attention = candidate.get("attention_score")
                    mood = candidate.get("emotional_tone")
                    st.markdown(f"**{index}. {title}**")
                    st.caption(
                        " | ".join(
                            part
                            for part in (
                                f"genres: {genres_text}" if genres_text else "",
                                f"themes: {themes}" if themes else "",
                                f"mood: {mood}" if mood else "",
                                f"rating: {rating}" if rating is not None else "",
                                f"attention: {attention}" if attention is not None else "",
                            )
                            if part
                        )
                    )
        if genres:
            with st.expander(f"Genre signals ({len(genres)})"):
                st.json(genres)
        return

    query_type = sql_results.get("query_type") or sql_results.get("hybrid_query_type")
    title = sql_results.get("title")
    label = "Structured SQL data"
    if query_type:
        label += f" - {query_type}"
    if title:
        label += f" - {title}"
    with st.expander(label):
        st.json(sql_results)


def _join_list(value: Any, limit: int) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    return ", ".join(str(item) for item in value[:limit])


def _snippet(text: Any, max_chars: int = 220) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rsplit(" ", 1)[0].rstrip(".,;:") + "..."


def _render_message(entry: dict[str, Any]) -> None:
    avatar = "🎬" if entry["role"] == "assistant" else "🙂"
    with st.chat_message(entry["role"], avatar=avatar):
        if entry.get("error"):
            st.error(entry["content"])
        else:
            st.markdown(entry["content"])
        if entry["role"] == "assistant" and entry.get("mode"):
            _render_meta(entry)
            _render_feedback(entry)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="sidebar-title">Cultural Mood Tracker</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sidebar-caption">A chat interface over the project\'s SQL + RAG orchestrator. '
        "Routing is deterministic: SQL handles facts, RAG handles text evidence, hybrid combines "
        "metrics with evidence, and recommendations are SQL-ranked with one review excerpt per title.</p>",
        unsafe_allow_html=True,
    )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []

    with st.expander("How answers are routed"):
        for mode_key in ("fast_sql", "rag", "hybrid", "recommendation"):
            info = MODE_INFO[mode_key]
            st.markdown(
                f'<div class="legend-row">{_mode_badge_html(mode_key)}'
                f'<span class="sidebar-caption">{info["description"]}</span></div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Backend connection (fails fast, with a readable message, instead of a raw traceback)
# ---------------------------------------------------------------------------
try:
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = get_orchestrator()
    orchestrator = st.session_state.orchestrator
    observability_store = get_observability_store()
except Exception as exc:  # noqa: BLE001 - deliberately broad: this is a top-level startup guard
    st.error(_friendly_error(exc))
    st.stop()


# ---------------------------------------------------------------------------
# Chat history + input
# ---------------------------------------------------------------------------
for entry in st.session_state.messages:
    _render_message(entry)

query = st.chat_input("Ask about a movie or TV show...")

if query:
    user_entry = {"role": "user", "content": query}
    st.session_state.messages.append(user_entry)
    _render_message(user_entry)

    with st.chat_message("assistant", avatar="🎬"):
        with st.spinner("Thinking..."):
            started_at = time.perf_counter()
            raw_error: str | None = None
            try:
                orchestrator.sql_client.clear_last_queries()
                result = orchestrator.answer(query)
                elapsed = time.perf_counter() - started_at
                sql_queries = orchestrator.sql_client.get_last_queries()
                evidence = [
                    {
                        "chunk_id": item.chunk_id,
                        "similarity": item.similarity,
                        "title": item.metadata.get("title_name"),
                        "source": item.metadata.get("source_name"),
                        "document_type": item.metadata.get("document_type") or item.metadata.get("chunk_source_type"),
                        "snippet": _snippet(item.chunk_text),
                        "lookup": "exact-title review" if result.mode == "recommendation" else None,
                    }
                    for item in result.retrieved_chunks
                ]
                assistant_entry = {
                    "role": "assistant",
                    "content": result.answer,
                    "mode": result.mode,
                    "used_sql": result.used_sql,
                    "elapsed": elapsed,
                    "sql_queries": sql_queries,
                    "sql_results": result.sql_results,
                    "evidence": evidence,
                    "error": False,
                }
            except Exception as exc:  # noqa: BLE001 - surfaced to the user as a chat bubble, not a crash
                raw_error = str(exc)
                elapsed = time.perf_counter() - started_at
                assistant_entry = {
                    "role": "assistant",
                    "content": _friendly_error(exc),
                    "mode": None,
                    "used_sql": False,
                    "elapsed": elapsed,
                    "sql_queries": [],
                    "sql_results": {},
                    "evidence": [],
                    "error": True,
                }

            evidence = assistant_entry.get("evidence") or []
            assistant_entry["interaction_id"] = observability_store.record_interaction(
                session_id=st.session_state.session_id,
                query=query,
                answer=assistant_entry["content"],
                mode=assistant_entry.get("mode"),
                latency_ms=(assistant_entry.get("elapsed") or 0.0) * 1000,
                error=bool(assistant_entry.get("error")),
                error_message=raw_error,
                retrieved_chunk_ids=[str(item.get("chunk_id")) for item in evidence],
                similarities=[
                    float(item["similarity"])
                    for item in evidence
                    if isinstance(item.get("similarity"), (int, float))
                ],
                model_name=orchestrator.llm_model_name,
            )

        if assistant_entry["error"]:
            st.error(assistant_entry["content"])
        else:
            st.markdown(assistant_entry["content"])
            _render_meta(assistant_entry)
            _render_feedback(assistant_entry)

    st.session_state.messages.append(assistant_entry)
