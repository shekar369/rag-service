# RAG Service

A production-minded Retrieval-Augmented Generation (RAG) service. It answers questions about your own documents with grounded, cited answers. It is built with **LangChain + FastAPI**, includes a **React chat UI**, and has observability, evaluation and monitoring built in.

```
 files --> load --> split --> upsert (Chroma, deterministic IDs)
                                   |
 question + history --> condense --> hybrid retrieve (MMR + BM25, RRF) --> grounded LLM answer + citations
        |                                                                       |
        +---- LangSmith traces / Prometheus metrics / JSON logs <---------------+
```

## Features

- **Idempotent ingestion**: PDF, Markdown and text files are chunked and upserted by content hash. Re-ingesting a changed file replaces its old chunks.
- **Hybrid retrieval**: Dense MMR search (Chroma) is fused with BM25 keyword search using Reciprocal Rank Fusion.
- **Conversational**: Follow-up questions are condensed with chat history into standalone queries.
- **Grounded answers with citations**: Each answer lists its sources. The service says "I don't know" instead of guessing.
- **Streaming**: Server-Sent Events stream sources first, then tokens.
- **Pluggable providers**: Choose NVIDIA, OpenAI, OpenAI-compatible (e.g. Ollama), Anthropic or Bedrock for the LLM, and NVIDIA, HuggingFace (local), OpenAI, Ollama or Bedrock for embeddings.
- **Observability**: Structured JSON logs, Prometheus metrics (latency per stage, tokens, unanswered ratio), LangSmith tracing and user feedback.
- **Evaluation gate**: A golden-set evaluator measures retrieval hit rate, faithfulness and correctness. It exits non-zero when a threshold is missed, so it can gate CI.
- **React UI**: A Vite + React 19 chat interface with Markdown rendering, source lists and file upload.

## Tech stack

| Layer         | Tools                                                        |
|---------------|--------------------------------------------------------------|
| API           | FastAPI, Uvicorn, Pydantic Settings                          |
| RAG           | LangChain (LCEL), Chroma, rank-bm25, pypdf                   |
| Models        | NVIDIA AI Endpoints, OpenAI, Anthropic, Bedrock, HuggingFace |
| Frontend      | React 19, Vite 7, react-markdown                             |
| Observability | LangSmith, Prometheus, Grafana                               |
| Tooling       | uv, ruff, pytest                                             |

## Quick start

### Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- Node.js 20+ (for the frontend)
- Docker (optional, for monitoring)

```bash
# Install uv
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 1. Install the backend

```bash
git clone https://github.com/shekar369/rag-service.git
cd rag-service

uv sync --group dev                       # core deps (NVIDIA, OpenAI, HuggingFace)
# Optional providers:
#   uv sync --extra anthropic --group dev
#   uv sync --extra bedrock
#   uv sync --all-extras --group dev
```

### 2. Configure

```bash
cp .env.example .env
```

Fill in `.env`:

| Variable             | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `NVIDIA_API_KEY`     | API key for NVIDIA AI Endpoints (when using the `nvidia` provider)          |
| `LLM_PROVIDER`       | `nvidia` \| `openai_compat` \| `ollama` \| `anthropic` \| `openai`          |
| `LLM_MODEL`          | Model name for the chosen provider                                          |
| `EMBEDDING_PROVIDER` | `huggingface` \| `nvidia` \| `openai_compat` \| `ollama` \| `openai` \| `bedrock` |
| `EMBEDDING_MODEL`    | Embedding model, e.g. `BAAI/bge-small-en-v1.5`                              |
| `HF_HUB_OFFLINE`     | `1` to load HuggingFace models from the local cache only                    |
| `LANGSMITH_TRACING`  | `true` to enable LangSmith traces                                           |
| `LANGSMITH_API_KEY`  | LangSmith API key                                                           |
| `LANGSMITH_PROJECT`  | LangSmith project name                                                      |

Other settings, such as chunk size, overlap, `top_k`, storage paths and the Ollama URL, have defaults in [app/config.py](app/config.py). You can override any of them in `.env`.

### 3. Ingest documents

```bash
# Put PDFs / .md / .txt files in ./data, then:
uv run python -m app.rag.ingest ./data
```

### 4. Run the API

```bash
uv run uvicorn app.main:app --reload --port 8000
```

OpenAPI docs: http://localhost:8000/docs

### 5. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The Vite dev server forwards `/api/*` to the FastAPI backend on port 8000.

## API

| Method | Endpoint       | Description                                          |
|--------|----------------|------------------------------------------------------|
| POST   | `/ingest`      | Upload and ingest files (multipart `files`)          |
| POST   | `/chat`        | Ask a question. Returns `answer`, `sources`, `run_id` |
| POST   | `/chat/stream` | Same as `/chat`, streamed as SSE (`sources`, `token`…, `done`) |
| POST   | `/feedback`    | Attach a score and comment to a LangSmith run        |
| GET    | `/metrics/`    | Prometheus metrics                                   |

```bash
curl -F "files=@handbook.pdf" localhost:8000/ingest

curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"question":"What is the leave policy?","session_id":"s1"}'

curl -N localhost:8000/chat/stream -H 'content-type: application/json' \
  -d '{"question":"And for contractors?","session_id":"s1"}'

curl -s localhost:8000/feedback -H 'content-type: application/json' \
  -d '{"run_id":"<run_id>","score":1,"comment":"helpful"}'
```

## Monitoring

```bash
cd monitoring && docker compose up -d
```

- Prometheus: http://localhost:9090 (alert rules are under Status > Rules)
- Grafana: http://localhost:3000. Add the data source `http://prometheus:9090`.

Useful queries:

- p95 latency by stage: `histogram_quantile(0.95, sum(rate(rag_stage_latency_seconds_bucket[5m])) by (le, stage))`
- Tokens per minute: `sum(rate(rag_llm_tokens_total[5m])) by (direction) * 60`
- "I don't know" ratio: `sum(rate(rag_unanswered_total[30m])) / sum(rate(rag_chat_total[30m]))`

## Evaluation

Add your questions to `eval/golden.jsonl`, one JSON object per line with `question`, `reference_answer` and `expected_sources`. Then run:

```bash
uv run python -m eval.run_eval --min-hit 0.8 --min-faith 4.0 --min-correct 3.5
```

Results are written to `eval/results.json`. The command returns a non-zero exit code when any threshold is missed.

## Development

```bash
uv add <pkg>                       # add a dependency
uv sync --frozen                   # install exactly what uv.lock specifies
uv run ruff check . && uv run ruff format .
uv run pytest -q
```

### Docker

```dockerfile
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY app ./app
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Project layout

```
app/
  config.py          settings (env driven)
  observability.py   JSON logs, Prometheus metrics, LangChain callback, LangSmith feedback
  main.py            FastAPI routes
  rag/
    models.py        LLM / embeddings / vector store factories
    ingest.py        load -> split -> upsert
    retriever.py     hybrid MMR + BM25 with RRF
    chain.py         LCEL chain: condense -> retrieve -> answer
frontend/            React + Vite chat UI
eval/                golden set + run_eval.py (CI gate)
monitoring/          prometheus.yml, alerts.yml, docker-compose.yml
data/                source documents (git-ignored)
```

## Author

**Shekar** · [github.com/shekar369](https://github.com/shekar369)
