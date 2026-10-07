import { useEffect, useRef, useState } from "react";
import { checkHealth } from "./api";
import Composer from "./components/Composer";
import MessageBubble from "./components/MessageBubble";
import UploadButton from "./components/UploadButton";
import { useChat } from "./hooks/useChat";
import "./styles.css";

const SUGGESTIONS = [
  "Summarize the main topics covered in the documents.",
  "What are the key definitions I should know?",
  "Which document discusses evaluation, and what does it say?",
];

export default function App() {
  const { messages, isStreaming, send, stop, rate, newChat } = useChat();
  const [online, setOnline] = useState(null);
  const listRef = useRef(null);
  const stickRef = useRef(true);

  useEffect(() => {
    checkHealth().then(setOnline);
  }, []);

  useEffect(() => {
    const el = listRef.current;
    if (el && stickRef.current) el.scrollTo({ top: el.scrollHeight });
  }, [messages]);

  const onScroll = () => {
    const el = listRef.current;
    stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  };

  return (
    <div className="app">
      <header className="header">
        <div className="title">
          <span className={`dot ${online === null ? "" : online ? "ok" : "bad"}`} />
          RAG Chat
        </div>
        <div className="header-actions">
          <UploadButton />
          <button className="btn ghost" onClick={newChat}>
            New chat
          </button>
        </div>
      </header>

      <main className="messages" ref={listRef} onScroll={onScroll}>
        {messages.length === 0 ? (
          <div className="empty">
            <h1>Ask your documents</h1>
            <p>Answers are grounded in your ingested files, with citations.</p>
            <div className="chips">
              {SUGGESTIONS.map((s) => (
                <button key={s} className="chip" onClick={() => send(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m) => <MessageBubble key={m.id} message={m} onRate={rate} />)
        )}
      </main>

      <footer className="footer">
        <Composer onSend={send} onStop={stop} isStreaming={isStreaming} />
        {online === false && (
          <div className="offline">Backend unreachable. Is FastAPI running on port 8000?</div>
        )}
      </footer>
    </div>
  );
}
