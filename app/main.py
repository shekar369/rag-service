"""Thin FastAPI wiring. All RAG logic lives in app/rag/*, so you can drop it
into your own FastAPI project by copying `lifespan` + the routes you need."""
import asyncio
import json
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field

from app.config import settings
from app.observability import (
    CHAT_TOTAL, ERRORS, FEEDBACK, HTTP_LATENCY, HTTP_REQUESTS, UNANSWERED,
    make_config, request_id_ctx, send_langsmith_feedback, setup_logging, tracing_enabled,
)
from app.rag.chain import build_chain
from app.rag.ingest import LOADERS, ingest_paths
from app.rag.models import get_llm, get_vectorstore, validate_nvidia_models
from app.rag.retriever import HybridRetriever

log = logging.getLogger("app")

# In-memory chat history. Swap for Redis/Postgres (or LangGraph checkpointer) later.
SESSIONS: dict[str, list[BaseMessage]] = defaultdict(list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    validate_nvidia_models()
    vs = get_vectorstore()
    retriever = HybridRetriever(vs)
    app.state.vs, app.state.retriever = vs, retriever
    app.state.chain = build_chain(get_llm(), retriever)
    log.info("startup complete", extra={"fields": {"tracing": tracing_enabled()}})
    yield


app = FastAPI(title="RAG Service", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def observe_http(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid4().hex
    request_id_ctx.set(rid)
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = rid
        return response
    finally:
        HTTP_REQUESTS.labels(request.method, request.url.path, str(status)).inc()
        HTTP_LATENCY.labels(request.url.path).observe(time.perf_counter() - t0)


# ---------- schemas ----------
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str = "default"


class Feedback(BaseModel):
    run_id: str
    score: int = Field(description="1 = thumbs up, -1 = thumbs down")
    comment: str | None = None


def _sources(docs) -> list[dict]:
    return [
        {
            "n": i,
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page"),
            "snippet": d.page_content[:200],
        }
        for i, d in enumerate(docs, 1)
    ]


def _remember(session_id: str, question: str, answer: str) -> None:
    h = SESSIONS[session_id]
    h += [HumanMessage(question), AIMessage(answer)]
    SESSIONS[session_id] = h[-2 * settings.max_history_turns :]


def _track_answer(answer: str) -> None:
    CHAT_TOTAL.inc()
    if "don't know" in answer.lower():  # cheap online quality proxy
        UNANSWERED.inc()


# ---------- routes ----------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(files: list[UploadFile]):
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in files:
        if Path(f.filename).suffix.lower() not in LOADERS:
            raise HTTPException(400, f"Unsupported file type: {f.filename}")
        dest = data_dir / Path(f.filename).name
        dest.write_bytes(await f.read())
        paths.append(dest)
    stats = await asyncio.to_thread(ingest_paths, app.state.vs, paths)
    await asyncio.to_thread(app.state.retriever.refresh)
    return stats


@app.post("/chat")
async def chat(req: ChatRequest):
    run_id = uuid4()
    cfg = make_config(run_id, req.session_id)
    try:
        out = await app.state.chain.ainvoke(
            {"question": req.question, "history": SESSIONS[req.session_id]}, config=cfg
        )
    except Exception:
        ERRORS.labels("chat").inc()
        log.exception("chat failed")
        raise HTTPException(500, "Failed to generate answer")
    _remember(req.session_id, req.question, out["answer"])
    _track_answer(out["answer"])
    log.info("chat", extra={"fields": {
        "run_id": str(run_id), "session_id": req.session_id,
        "standalone": out["standalone"], "n_docs": len(out["docs"]),
    }})
    return {"run_id": str(run_id), "answer": out["answer"], "sources": _sources(out["docs"])}


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    run_id = uuid4()
    cfg = make_config(run_id, req.session_id, extra_tags=["stream"])
    history = SESSIONS[req.session_id]

    async def gen():
        tokens: list[str] = []
        try:
            async for chunk in app.state.chain.astream(
                {"question": req.question, "history": history}, config=cfg
            ):
                if "docs" in chunk:
                    yield _sse("sources", _sources(chunk["docs"]))
                if "answer" in chunk:
                    tokens.append(chunk["answer"])
                    yield _sse("token", chunk["answer"])
        except Exception:
            ERRORS.labels("chat_stream").inc()
            log.exception("stream failed")
            yield _sse("error", "generation failed")
            return
        answer = "".join(tokens)
        _remember(req.session_id, req.question, answer)
        _track_answer(answer)
        yield _sse("done", {"run_id": str(run_id)})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/feedback")
async def feedback(fb: Feedback):
    FEEDBACK.labels("up" if fb.score > 0 else "down").inc()
    if tracing_enabled():
        try:
            await asyncio.to_thread(send_langsmith_feedback, fb.run_id, float(fb.score), fb.comment)
        except Exception:
            log.exception("langsmith feedback failed")
    return {"ok": True}