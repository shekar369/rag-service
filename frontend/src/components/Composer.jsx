import { useEffect, useRef, useState } from "react";

export default function Composer({ onSend, onStop, isStreaming, left }) {
  const [text, setText] = useState("");
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, [text]);

  const submit = () => {
    if (!text.trim() || isStreaming) return;
    onSend(text);
    setText("");
  };

  return (
    <div className="composer">
      {left}
      <textarea
        ref={ref}
        rows={1}
        value={text}
        placeholder="Ask about your documents…"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            submit();
          }
        }}
      />
      {isStreaming ? (
        <button className="btn stop" onClick={onStop}>
          Stop
        </button>
      ) : (
        <button className="btn send" onClick={submit} disabled={!text.trim()}>
          Send
        </button>
      )}
    </div>
  );
}
