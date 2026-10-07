import { useCallback, useEffect, useRef, useState } from "react";
import { sendFeedback, streamChat } from "../api";

const SESSION_KEY = "rag.session";
const MSGS_KEY = "rag.messages";

const newId = () =>
  globalThis.crypto?.randomUUID?.() ??
  `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;

function load(key, fallback) {
  try {
    const v = localStorage.getItem(key);
    return v ? JSON.parse(v) : fallback;
  } catch {
    return fallback;
  }
}

export function useChat() {
  const [sessionId, setSessionId] = useState(() => load(SESSION_KEY, null) || newId());
  const [messages, setMessages] = useState(() =>
    load(MSGS_KEY, []).map((m) => (m.status === "streaming" ? { ...m, status: "stopped" } : m))
  );
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef(null);

  useEffect(() => {
    localStorage.setItem(SESSION_KEY, JSON.stringify(sessionId));
  }, [sessionId]);

  useEffect(() => {
    if (!isStreaming) localStorage.setItem(MSGS_KEY, JSON.stringify(messages));
  }, [messages, isStreaming]);

  const patch = useCallback((id, fn) => {
    setMessages((ms) => ms.map((m) => (m.id === id ? { ...m, ...fn(m) } : m)));
  }, []);

  const send = useCallback(
    async (text) => {
      const question = text.trim();
      if (!question || isStreaming) return;

      const botId = newId();
      setMessages((ms) => [
        ...ms,
        { id: newId(), role: "user", content: question },
        { id: botId, role: "assistant", content: "", sources: [], status: "streaming" },
      ]);
      setIsStreaming(true);
      const ctrl = new AbortController();
      abortRef.current = ctrl;

      try {
        for await (const { event, data } of streamChat({
          question,
          sessionId,
          signal: ctrl.signal,
        })) {
          if (event === "sources") patch(botId, () => ({ sources: data }));
          else if (event === "token") patch(botId, (m) => ({ content: m.content + data }));
          else if (event === "done") patch(botId, () => ({ runId: data.run_id, status: "done" }));
          else if (event === "error") patch(botId, () => ({ status: "error", error: String(data) }));
        }
        patch(botId, (m) => (m.status === "streaming" ? { status: "done" } : {}));
      } catch (e) {
        if (e.name === "AbortError") patch(botId, () => ({ status: "stopped" }));
        else patch(botId, () => ({ status: "error", error: e.message }));
      } finally {
        setIsStreaming(false);
        abortRef.current = null;
      }
    },
    [isStreaming, sessionId, patch]
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  const rate = useCallback(
    async (id, score) => {
      const msg = messages.find((m) => m.id === id);
      if (!msg?.runId) return;
      const prev = msg.feedback ?? null;
      const next = prev === score ? null : score;
      patch(id, () => ({ feedback: next }));
      if (next === null) return; // backend has no "undo"; clearing is UI-only
      try {
        await sendFeedback({ runId: msg.runId, score: next });
      } catch {
        patch(id, () => ({ feedback: prev }));
      }
    },
    [messages, patch]
  );

  const newChat = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setSessionId(newId()); // new session => fresh server-side history
  }, []);

  return { messages, isStreaming, send, stop, rate, newChat };
}
