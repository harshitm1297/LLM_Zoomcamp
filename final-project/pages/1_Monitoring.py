from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cultural_mood_tracker.config import load_settings
from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.observability import ObservabilityStore


st.set_page_config(page_title="Cultural Mood Tracker Monitoring", page_icon="📊", layout="wide")
st.title("Application Monitoring")

project_root = load_project_environment(Path(__file__))
settings = load_settings()
database_path = settings.observability_db_path
if not database_path.is_absolute():
    database_path = project_root / database_path
store = ObservabilityStore(database_path)
summary = store.summary()
rows = store.interactions()

columns = st.columns(5)
columns[0].metric("Requests", int(summary.get("requests") or 0))
columns[1].metric("Errors", int(summary.get("errors") or 0))
columns[2].metric("Feedback responses", int(summary.get("responses") or 0))
columns[3].metric("Positive feedback", f"{100 * float(summary.get('positive_rate') or 0):.1f}%")
columns[4].metric("Mean latency", f"{float(summary.get('average_latency_ms') or 0):.0f} ms")

if not rows:
    st.info("No interactions have been recorded yet. Use the chat application, then refresh this page.")
    st.stop()

frame = pd.DataFrame(rows)
frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True)
frame["date"] = frame["created_at"].dt.date
frame["mode"] = frame["mode"].fillna("error")

st.subheader("1. Requests over time")
st.line_chart(frame.groupby("date").size().rename("requests"))

st.subheader("2. Positive feedback rate over time")
feedback = frame.dropna(subset=["feedback_score"]).copy()
if feedback.empty:
    st.info("No feedback recorded yet.")
else:
    feedback["positive"] = (feedback["feedback_score"] == 1).astype(float)
    st.line_chart(feedback.groupby("date")["positive"].mean())

left, right = st.columns(2)
with left:
    st.subheader("3. Mean latency by route")
    st.bar_chart(frame.groupby("mode")["latency_ms"].mean())
with right:
    st.subheader("4. Route distribution")
    st.bar_chart(frame.groupby("mode").size().rename("requests"))

left, right = st.columns(2)
with left:
    st.subheader("5. Errors over time")
    st.line_chart(frame.groupby("date")["error"].sum())
with right:
    st.subheader("6. Mean retrieval similarity by route")
    similarity = frame.dropna(subset=["average_similarity"])
    if similarity.empty:
        st.info("No retrieval similarities recorded yet.")
    else:
        st.bar_chart(similarity.groupby("mode")["average_similarity"].mean())

st.subheader("7. Recent interactions")
st.dataframe(
    frame[["created_at", "mode", "query", "latency_ms", "error", "feedback_score"]]
    .sort_values("created_at", ascending=False)
    .head(100),
    use_container_width=True,
)
