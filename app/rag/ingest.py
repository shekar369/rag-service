"""Ingestion: load -> split -> stamp metadata -> idempotent upsert.

Re-ingesting a file first deletes its old chunks (by `source`), so edits never
leave stale chunks behind. Chunk IDs are deterministic content hashes.

CLI:  python -m app.rag.ingest ./data
"""
import hashlib
import logging
import sys
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

log = logging.getLogger(__name__)

LOADERS = {".pdf": PyPDFLoader, ".md": TextLoader, ".txt": TextLoader}
BATCH = 100


def load_paths(paths: list[Path]) -> list[Document]:
    docs: list[Document] = []
    for p in paths:
        loader_cls = LOADERS.get(p.suffix.lower())
        if loader_cls is None:
            log.warning("skip unsupported file %s", p)
            continue
        kwargs = {"encoding": "utf-8"} if loader_cls is TextLoader else {}
        docs.extend(loader_cls(str(p), **kwargs).load())
    return docs


def chunk(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    now = int(time.time())
    for c in chunks:
        c.metadata["source"] = Path(c.metadata.get("source", "unknown")).name
        raw = (
            f'{c.metadata["source"]}:{c.metadata.get("page", "")}:'
            f'{c.metadata["start_index"]}:{c.page_content}'
        )
        c.metadata["chunk_id"] = hashlib.sha1(raw.encode()).hexdigest()
        c.metadata["ingested_at"] = now
    return chunks


def ingest_paths(vs, paths: list[Path]) -> dict:
    t0 = time.perf_counter()
    docs = load_paths(paths)
    chunks = chunk(docs)

    for name in {c.metadata["source"] for c in chunks}:
        vs._collection.delete(where={"source": name})  # drop stale chunks

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        vs.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])

    stats = {
        "files": len(paths),
        "pages_or_docs": len(docs),
        "chunks": len(chunks),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    log.info("ingest complete", extra={"fields": stats})
    return stats


if __name__ == "__main__":
    from app.observability import setup_logging
    from app.rag.models import get_vectorstore

    setup_logging()
    root = Path(sys.argv[1] if len(sys.argv) > 1 else settings.data_dir)
    files = [p for p in root.rglob("*") if p.suffix.lower() in LOADERS]
    print(ingest_paths(get_vectorstore(), files))
