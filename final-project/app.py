"""High-contrast Streamlit frontend for the RAG-only chatbot."""

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

from cultural_mood_tracker.chat import ChatOrchestrator
from cultural_mood_tracker.config import load_settings
from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.observability import ObservabilityStore
from cultural_mood_tracker.rag.chroma_ingest import (
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_CHROMA_DB_DIR,
)
from cultural_mood_tracker.rag.embeddings import DEFAULT_EMBEDDING_MODEL
from cultural_mood_tracker.rag.llm import DEFAULT_MODEL
from cultural_mood_tracker.rag.prompting import DEFAULT_MAX_CONTEXT_CHARS


st.set_page_config(
    page_title="Pop Culture Detective",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded",
)

APP_CSS = """
<style>
:root {
    --ink: #172033;
    --muted: #5f6b7a;
    --line: #dbe2ea;
    --surface: #ffffff;
    --page: #f4f7fb;
    --primary: #3157d5;
    --primary-dark: #203b96;
    --user: #e7efff;
    --assistant: #ffffff;
}

.stApp {
    background: var(--page);
    color: var(--ink);
}
.block-container {
    max-width: 900px;
    padding-top: 1.8rem;
    padding-bottom: 7rem;
}
html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.app-hero {
    padding: 1.65rem 1.8rem;
    margin-bottom: 1rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #203b96 0%, #3157d5 58%, #5878df 100%);
    box-shadow: 0 12px 30px rgba(32, 59, 150, 0.18);
}
.app-kicker {
    margin: 0 0 0.35rem 0;
    color: #dce6ff;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.app-title {
    margin: 0;
    color: #ffffff;
    font-size: 2rem;
    font-weight: 750;
    letter-spacing: -0.025em;
}
.app-title,
.app-title a,
.app-title span {
    color: #ffffff !important;
}
.app-subtitle {
    margin: 0.45rem 0 0 0;
    color: #eef3ff;
    font-size: 0.98rem;
    line-height: 1.55;
}
.example-strip {
    color: var(--muted);
    font-size: 0.86rem;
    margin: 0 0 1rem 0;
}

[data-testid="stChatMessage"] {
    background: var(--assistant);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 0.45rem 0.65rem;
    box-shadow: 0 3px 12px rgba(23, 32, 51, 0.045);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--user);
    border-color: #c7d7ff;
}
[data-testid="stChatMessage"][aria-label="Chat message from user"] {
    background: var(--user);
    border-color: #c7d7ff;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    color: var(--ink);
}
[data-testid="stChatMessage"] code {
    color: #172033;
    background: #edf1f6;
}
[data-testid="stChatInput"] textarea {
    color: var(--ink) !important;
    background: #ffffff !important;
}
[data-testid="stChatInput"] {
    border: 1px solid #cbd5e1;
    box-shadow: 0 6px 24px rgba(23, 32, 51, 0.10);
}

.answer-meta {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
    margin: 0.6rem 0 0.15rem 0;
}
.meta-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.62rem;
    border-radius: 999px;
    border: 1px solid #ccd6e4;
    background: #f6f8fb;
    color: #455268 !important;
    font-size: 0.72rem;
    font-weight: 650;
}
.rag-pill {
    border-color: #b8c9ff;
    background: #eaf0ff;
    color: var(--primary-dark) !important;
}
.evidence-note {
    color: var(--muted) !important;
    font-size: 0.84rem;
    line-height: 1.45;
}

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] * {
    color: var(--ink);
}
.sidebar-title {
    margin: 0;
    color: var(--primary-dark) !important;
    font-size: 1.25rem;
    font-weight: 750;
}
.sidebar-copy {
    color: var(--muted) !important;
    font-size: 0.88rem;
    line-height: 1.5;
}
.status-card {
    padding: 0.8rem 0.9rem;
    margin: 0.7rem 0 1rem 0;
    border: 1px solid #cfe0d6;
    border-radius: 12px;
    background: #f0faf4;
    color: #235c3b !important;
    font-size: 0.83rem;
    line-height: 1.45;
}

.stButton > button {
    border-radius: 10px;
    border-color: #cbd5e1;
    color: var(--ink);
    background: #ffffff;
}
.stButton > button:hover {
    border-color: var(--primary);
    color: var(--primary-dark);
}
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="app-hero">
        <p class="app-kicker">Local-first RAG assistant</p>
        <h1 class="app-title">Pop Culture Detective</h1>
        <p class="app-subtitle">Explore movie and television descriptions with answers grounded only in the indexed source passages.</p>
    </div>
    <p class="example-strip"><strong>Try:</strong> What happens in Disclosure Day? &nbsp;·&nbsp; What themes appear in Obsession? &nbsp;·&nbsp; Which title involves a scientist in space?</p>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_orchestrator() -> ChatOrchestrator:
    project_root = load_project_environment(Path(__file__))
    settings = load_settings()
    persist_dir = Path(DEFAULT_CHROMA_DB_DIR)
    if not persist_dir.is_absolute():
        persist_dir = project_root / persist_dir
    chatbot = ChatOrchestrator(
        persist_dir=persist_dir,
        collection_name=DEFAULT_CHROMA_COLLECTION,
        embedding_model_name=DEFAULT_EMBEDDING_MODEL,
        llm_model_name=DEFAULT_MODEL,
        top_k=settings.retrieval_top_k,
        max_context_chars=DEFAULT_MAX_CONTEXT_CHARS,
    )
    chatbot.healthcheck()
    return chatbot


@st.cache_resource(show_spinner=False)
def get_observability_store() -> ObservabilityStore:
    project_root = load_project_environment(Path(__file__))
    settings = load_settings()
    path = settings.observability_db_path
    if not path.is_absolute():
        path = project_root / path
    return ObservabilityStore(path)


def _friendly_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.casefold()
    if "groq_api_key" in lowered or "groq generation failed" in lowered:
        return "The LLM call failed. Check that `GROQ_API_KEY` is present and valid."
    if "chroma" in lowered or "collection" in lowered or "persist" in lowered:
        return "The vector knowledge base is unavailable. Run `python scripts/bootstrap.py --sample`."
    return f"The answer could not be generated: {message}"


def _snippet(text: Any, max_chars: int = 240) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rsplit(" ", 1)[0].rstrip(".,;:") + "..."


def _render_answer_meta(entry: dict[str, Any]) -> None:
    evidence = entry.get("evidence") or []
    elapsed = float(entry.get("elapsed") or 0.0)
    st.markdown(
        '<div class="answer-meta">'
        '<span class="meta-pill rag-pill">RAG answer</span>'
        f'<span class="meta-pill">{len(evidence)} sources</span>'
        f'<span class="meta-pill">{elapsed:.2f}s</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    if evidence:
        with st.expander(f"Retrieved evidence ({len(evidence)})"):
            for index, item in enumerate(evidence, start=1):
                title = item.get("title") or "Unknown title"
                source = item.get("source") or "unknown source"
                similarity = item.get("similarity")
                score = f" · similarity {similarity:.3f}" if isinstance(similarity, (int, float)) else ""
                st.markdown(f"**{index}. {title}** — {source}{score}")
                st.caption(_snippet(item.get("snippet")))


def _render_feedback(entry: dict[str, Any]) -> None:
    interaction_id = entry.get("interaction_id")
    if not interaction_id:
        return
    selection = st.feedback("thumbs", key=f"feedback_{interaction_id}")
    if selection is not None:
        score = 1 if selection == 1 else -1
        get_observability_store().record_feedback(interaction_id, score)
        st.caption("Feedback saved. Thank you.")


def _render_message(entry: dict[str, Any]) -> None:
    avatar = "🎬" if entry["role"] == "assistant" else "🙂"
    with st.chat_message(entry["role"], avatar=avatar):
        if entry.get("error"):
            st.error(entry["content"])
        else:
            st.markdown(entry["content"])
        if entry["role"] == "assistant" and not entry.get("error"):
            _render_answer_meta(entry)
            _render_feedback(entry)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

try:
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = get_orchestrator()
    orchestrator = st.session_state.orchestrator
    observability_store = get_observability_store()
    health = orchestrator.healthcheck()
except Exception as exc:  # noqa: BLE001
    st.error(_friendly_error(exc))
    st.stop()

with st.sidebar:
    st.markdown('<p class="sidebar-title">About this assistant</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sidebar-copy">Every question follows one path: semantic vector retrieval from Chroma, a grounded prompt, and a Groq answer. SQL and hybrid answer modes are not used.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="status-card">● Knowledge base ready<br>{health["chunk_count"]} indexed passages · vector retrieval</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sidebar-copy"><strong>Bundled demo:</strong> 8 titles — 6 movies and 2 TV series. Run the optional full ingestion pipeline to build a larger corpus.</p>',
        unsafe_allow_html=True,
    )
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    _render_message(message)

query = st.chat_input("Ask about the indexed movies and TV shows...")
if query:
    user_entry = {"role": "user", "content": query}
    st.session_state.messages.append(user_entry)
    _render_message(user_entry)

    with st.chat_message("assistant", avatar="🎬"):
        with st.spinner("Retrieving evidence and writing an answer..."):
            started_at = time.perf_counter()
            raw_error: str | None = None
            try:
                result = orchestrator.answer(query)
                elapsed = time.perf_counter() - started_at
                evidence = [
                    {
                        "chunk_id": chunk.chunk_id,
                        "similarity": chunk.similarity,
                        "title": chunk.metadata.get("title_name"),
                        "source": chunk.metadata.get("source_name"),
                        "document_type": chunk.metadata.get("document_type"),
                        "snippet": chunk.chunk_text,
                    }
                    for chunk in result.retrieved_chunks
                ]
                assistant_entry = {
                    "role": "assistant",
                    "content": result.answer,
                    "mode": "rag",
                    "elapsed": elapsed,
                    "evidence": evidence,
                    "error": False,
                }
            except Exception as exc:  # noqa: BLE001
                raw_error = str(exc)
                elapsed = time.perf_counter() - started_at
                assistant_entry = {
                    "role": "assistant",
                    "content": _friendly_error(exc),
                    "mode": "rag",
                    "elapsed": elapsed,
                    "evidence": [],
                    "error": True,
                }

            evidence = assistant_entry["evidence"]
            assistant_entry["interaction_id"] = observability_store.record_interaction(
                session_id=st.session_state.session_id,
                query=query,
                answer=assistant_entry["content"],
                mode="rag",
                latency_ms=elapsed * 1000,
                error=assistant_entry["error"],
                error_message=raw_error,
                retrieved_chunk_ids=[str(item["chunk_id"]) for item in evidence],
                similarities=[float(item["similarity"]) for item in evidence],
                model_name=orchestrator.llm_model_name,
            )

        if assistant_entry["error"]:
            st.error(assistant_entry["content"])
        else:
            st.markdown(assistant_entry["content"])
            _render_answer_meta(assistant_entry)
            _render_feedback(assistant_entry)

    st.session_state.messages.append(assistant_entry)
