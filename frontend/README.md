# RAG Chat UI (React + Vite)

Streaming chat UI for the FastAPI RAG service: token streaming, clickable
`[n]` citations with a sources panel, thumbs up/down (feeds LangSmith +
Prometheus), file upload to `/ingest`, stop button, persistent session.

## Run (PowerShell)

Needs Node 20.19+ (or 22.12+).

```powershell
# terminal 1: backend
cd C:\Users\Admin\Documents\projects\rag-service
uv run uvicorn app.main:app --reload --port 8000

# terminal 2: frontend
cd C:\Users\Admin\Documents\projects\rag-service\frontend
npm install
npm run dev
```

Open http://localhost:5173. In dev, Vite proxies `/api/*` to `localhost:8000`,
so no CORS setup is needed.

## Production options

**A. Serve the build from FastAPI (single origin, no CORS).**
```powershell
cd frontend; npm run build
```
Then at the very end of `app/main.py`:
```python
from fastapi.staticfiles import StaticFiles
# API routes must be registered above this line
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="ui")
```
Build with `VITE_API_BASE=""` so calls go to `/chat/stream` instead of `/api/chat/stream`:
```powershell
$env:VITE_API_BASE=""; npm run build
```

**B. Separate origin (nginx / S3 / Vercel).** Build with
`$env:VITE_API_BASE="https://api.example.com"` and enable CORS in `app/main.py`:
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-ui-origin"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```
If you proxy with nginx, disable buffering for SSE: `proxy_buffering off;`.

## Structure
```
src/
  api.js                  fetch + SSE parser (POST streaming), feedback, ingest, health
  hooks/useChat.js        messages, streaming, abort, feedback, new chat, localStorage
  components/
    MessageBubble.jsx     markdown, citation chips, sources, feedback
    SourceList.jsx        collapsible sources with highlight
    Composer.jsx          auto-grow textarea, Enter to send, Stop button
    UploadButton.jsx      multi-file upload to /ingest
  App.jsx, styles.css     layout, health dot, empty state, auto-scroll
```
