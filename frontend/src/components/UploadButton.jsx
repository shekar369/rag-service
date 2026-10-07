import { useRef, useState } from "react";
import { ingestFiles } from "../api";

export default function UploadButton() {
  const inputRef = useRef(null);
  const [status, setStatus] = useState(null); // { kind: "busy" | "ok" | "err", text }

  const onPick = async (e) => {
    const files = Array.from(e.target.files || []);
    e.target.value = "";
    if (!files.length) return;
    setStatus({ kind: "busy", text: `Ingesting ${files.length} file(s)…` });
    try {
      const s = await ingestFiles(files);
      setStatus({ kind: "ok", text: `Indexed ${s.chunks} chunks in ${s.seconds}s` });
    } catch (err) {
      setStatus({ kind: "err", text: err.message });
    }
    setTimeout(() => setStatus(null), 6000);
  };

  return (
    <div className="upload">
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".pdf,.md,.txt"
        hidden
        onChange={onPick}
      />
      <button
        className="btn ghost"
        onClick={() => inputRef.current?.click()}
        disabled={status?.kind === "busy"}
        title="Upload PDF / MD / TXT"
      >
        Upload
      </button>
      {status && <span className={`upload-status ${status.kind}`}>{status.text}</span>}
    </div>
  );
}
