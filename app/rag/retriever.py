"""Hybrid retrieval: dense (MMR) + BM25, fused with Reciprocal Rank Fusion.

Implemented as a RunnableLambda so LangChain callbacks / LangSmith traces show
the dense and sparse retrievers as child runs of `hybrid_retrieve`.
"""
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig, RunnableLambda

from app.config import settings


def rrf(result_lists: list[list[Document]], k: int = 60, top_n: int = 6) -> list[Document]:
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for results in result_lists:
        for rank, d in enumerate(results):
            key = d.metadata.get("chunk_id") or str(hash(d.page_content))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            docs[key] = d
    ordered = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return [docs[key] for key in ordered]


class HybridRetriever:
    def __init__(self, vs, k: int = settings.top_k):
        self.vs, self.k = vs, k
        self.dense = vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 4, "lambda_mult": 0.6},
        )
        self.sparse: BM25Retriever | None = None
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the in-memory BM25 index. Call after every ingest.
        (Fine up to ~100k chunks; beyond that use OpenSearch / pgvector FTS.)"""
        data = self.vs.get(include=["documents", "metadatas"])
        docs = [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(data["documents"], data["metadatas"])
        ]
        self.sparse = BM25Retriever.from_documents(docs, k=self.k) if docs else None

    def _retrieve(self, query: str, config: RunnableConfig) -> list[Document]:
        lists = [self.dense.invoke(query, config=config)]
        if self.sparse is not None:
            lists.append(self.sparse.invoke(query, config=config))
        return rrf(lists, top_n=self.k)

    @property
    def runnable(self) -> RunnableLambda:
        return RunnableLambda(self._retrieve, name="hybrid_retrieve")
