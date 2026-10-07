// Dev: "/api" is proxied by Vite. Prod on another origin: set VITE_API_BASE
// (and enable CORS on FastAPI).
const BASE = import.meta.env.VITE_API_BASE ?? "/api";

function parseBlock(block) {
  let event = "message";
  const dataLines = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  if (!dataLines.length) return null;
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return null;
  }
}

/**
 * POST + SSE (EventSource only supports GET, so we parse the stream ourselves).
 * Yields { event: "sources" | "token" | "done" | "error", data }.
 */
export async function* streamChat({ question, sessionId, signal }) {
  const res = await fetch(`${BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: sessionId }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(`Request failed (${res.status})`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = buffer.replace(/\r\n/g, "\n");
    let idx;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const evt = parseBlock(buffer.slice(0, idx));
      buffer = buffer.slice(idx + 2);
      if (evt) yield evt;
    }
  }
}

export async function sendFeedback({ runId, score, comment }) {
  const res = await fetch(`${BASE}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id: runId, score, comment }),
  });
  if (!res.ok) throw new Error(`Feedback failed (${res.status})`);
}

export async function ingestFiles(files) {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  const res = await fetch(`${BASE}/ingest`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `Ingest failed (${res.status})`);
  }
  return res.json(); // { files, pages_or_docs, chunks, seconds }
}

export async function checkHealth() {
  try {
    const res = await fetch(`${BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
