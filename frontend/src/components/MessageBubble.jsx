import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import SourceList from "./SourceList";

// "[1]" -> "[1](#cite-1)" so citations become clickable chips (skip real md links).
const withCitations = (t) => t.replace(/\[(\d+)\](?!\()/g, "[$1](#cite-$1)");

export default function MessageBubble({ message, onRate }) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(null);

  if (message.role === "user") {
    return (
      <div className="row user">
        <div className="bubble user-bubble">{message.content}</div>
      </div>
    );
  }

  const { content, sources, status, error, feedback, runId } = message;
  const streaming = status === "streaming";
  const phase = !sources?.length ? "Searching documents…" : "Generating answer…";

  const components = {
    table: ({ children }) => (
      <div className="table-wrap">
        <table>{children}</table>
      </div>
    ),
    a: ({ href, children }) => {
      const m = /^#cite-(\d+)$/.exec(href || "");
      if (m) {
        return (
          <button
            className="cite"
            title="Show source"
            onClick={() => {
              setOpen(true);
              setActive(Number(m[1]));
            }}
          >
            {m[1]}
          </button>
        );
      }
      return (
        <a href={href} target="_blank" rel="noreferrer">
          {children}
        </a>
      );
    },
  };

  return (
    <div className="row bot">
      <div className="avatar">AI</div>
      <div className="bubble bot-bubble">
        {streaming && !content && <div className="phase">{phase}</div>}
        {content && (
          <div className={streaming ? "md streaming" : "md"}>
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>{withCitations(content)}</ReactMarkdown>
          </div>
        )}
        {status === "stopped" && <div className="note">Generation stopped.</div>}
        {status === "error" && <div className="error">Error: {error || "something went wrong"}</div>}

        <SourceList
          sources={sources}
          open={open}
          active={active}
          onToggle={() => setOpen((o) => !o)}
        />

        {status === "done" && runId && (
          <div className="feedback">
            <button
              className={feedback === 1 ? "fb on" : "fb"}
              onClick={() => onRate(message.id, 1)}
              aria-label="Helpful"
            >
              👍
            </button>
            <button
              className={feedback === -1 ? "fb on" : "fb"}
              onClick={() => onRate(message.id, -1)}
              aria-label="Not helpful"
            >
              👎
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
