"""Observability: structured JSON logs, Prometheus metrics, LangChain callback.

Three layers:
  1. LangSmith  - per-request traces (enabled purely via LANGSMITH_* env vars)
  2. Prometheus - aggregate latency / tokens / quality-proxy metrics (/metrics)
  3. JSON logs  - request-id correlated, ship to CloudWatch / Loki / ELK
"""
import contextvars
import json
import logging
import os
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from prometheus_client import Counter, Histogram

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# ---------- logging ----------
_STD = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "fields"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_ctx.get(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


# ---------- metrics ----------
LAT_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60)

HTTP_REQUESTS = Counter("rag_http_requests_total", "HTTP requests", ["method", "path", "status"])
HTTP_LATENCY = Histogram("rag_http_latency_seconds", "HTTP latency", ["path"], buckets=LAT_BUCKETS)
STAGE_LATENCY = Histogram("rag_stage_latency_seconds", "Pipeline stage latency", ["stage"], buckets=LAT_BUCKETS)
DOCS_RETRIEVED = Histogram("rag_docs_retrieved", "Docs returned by retriever", buckets=(0, 1, 2, 4, 6, 8, 12))
LLM_TOKENS = Counter("rag_llm_tokens_total", "LLM tokens", ["direction"])
CHAT_TOTAL = Counter("rag_chat_total", "Chat answers produced")
UNANSWERED = Counter("rag_unanswered_total", "Answers where model said it doesn't know")
FEEDBACK = Counter("rag_feedback_total", "User feedback", ["vote"])
ERRORS = Counter("rag_errors_total", "Pipeline errors", ["where"])

_TRACKED = {"hybrid_retrieve", "standalone_question", "generate_answer"}


class MetricsCallback(BaseCallbackHandler):
    """Per-request callback: stage latency, retrieval size, token usage."""

    def __init__(self) -> None:
        self._t: dict[UUID, tuple[str, float]] = {}

    # chains / runnables (retrieval is a RunnableLambda named hybrid_retrieve)
    def on_chain_start(self, serialized, inputs, *, run_id: UUID, **kwargs: Any) -> None:
        name = kwargs.get("name")
        if name in _TRACKED:
            self._t[run_id] = (name, time.perf_counter())

    def on_chain_end(self, outputs, *, run_id: UUID, **kwargs: Any) -> None:
        entry = self._t.pop(run_id, None)
        if not entry:
            return
        stage, t0 = entry
        STAGE_LATENCY.labels(stage).observe(time.perf_counter() - t0)
        if stage == "hybrid_retrieve" and isinstance(outputs, list):
            DOCS_RETRIEVED.observe(len(outputs))

    def on_chain_error(self, error, *, run_id: UUID, **kwargs: Any) -> None:
        entry = self._t.pop(run_id, None)
        ERRORS.labels(entry[0] if entry else "chain").inc()

    # LLM calls
    def on_chat_model_start(self, serialized, messages, *, run_id: UUID, **kwargs: Any) -> None:
        self._t[run_id] = ("llm", time.perf_counter())

    def on_llm_end(self, response, *, run_id: UUID, **kwargs: Any) -> None:
        entry = self._t.pop(run_id, None)
        if entry:
            STAGE_LATENCY.labels("llm").observe(time.perf_counter() - entry[1])
        try:
            usage = getattr(response.generations[0][0].message, "usage_metadata", None) or {}
            LLM_TOKENS.labels("input").inc(usage.get("input_tokens", 0))
            LLM_TOKENS.labels("output").inc(usage.get("output_tokens", 0))
        except Exception:  # never let metrics break a request
            pass

    def on_llm_error(self, error, *, run_id: UUID, **kwargs: Any) -> None:
        self._t.pop(run_id, None)
        ERRORS.labels("llm").inc()


def make_config(run_id: UUID, session_id: str, extra_tags: list[str] | None = None) -> dict:
    """RunnableConfig for one request: LangSmith run id/tags/metadata + metrics."""
    return {
        "run_id": run_id,  # returned to client so feedback can attach to this trace
        "run_name": "rag_chat",
        "tags": ["rag", *(extra_tags or [])],
        "metadata": {"session_id": session_id, "request_id": request_id_ctx.get()},
        "callbacks": [MetricsCallback()],
    }


def tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true"


def send_langsmith_feedback(run_id: str, score: float, comment: str | None) -> None:
    from langsmith import Client

    Client().create_feedback(run_id=run_id, key="user_score", score=score, comment=comment)
