export default function SourceList({ sources, open, onToggle, active }) {
  if (!sources?.length) return null;
  return (
    <div className="sources">
      <button className="sources-toggle" onClick={onToggle}>
        {open ? "Hide" : "Show"} sources ({sources.length})
      </button>
      {open && (
        <ol className="sources-list">
          {sources.map((s) => (
            <li key={s.n} className={active === s.n ? "source active" : "source"}>
              <div className="source-head">
                <span className="source-n">{s.n}</span>
                <span className="source-name">
                  {s.source}
                  {s.page != null ? ` · p.${Number(s.page) + 1}` : ""}
                </span>
              </div>
              <p className="source-snippet">{s.snippet}…</p>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
