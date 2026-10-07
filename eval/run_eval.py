"""Offline evaluation + CI gate.

Metrics per question:
  - hit@k          : did retrieval return an expected source? (no LLM needed)
  - faithfulness   : is the answer supported by the retrieved context? (LLM judge, 1-5)
  - correctness    : does the answer match the reference answer? (LLM judge, 1-5)
  - latency

Usage:
  python -m eval.run_eval --golden eval/golden.jsonl --min-hit 0.8 --min-faith 4.0 --min-correct 3.5
Exits non-zero if any threshold is missed, so it can gate CI.

Upgrade path: same dataset can be pushed to LangSmith and run with
langsmith.evaluate(...), or swap the judges for RAGAS metrics.
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.rag.chain import build_chain, format_docs
from app.rag.models import get_llm, get_vectorstore
from app.rag.retriever import HybridRetriever


class Judgement(BaseModel):
    faithfulness: int = Field(ge=1, le=5, description="5 = every claim supported by context")
    correctness: int = Field(ge=1, le=5, description="5 = matches the reference answer")
    reasoning: str


JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict evaluator of a RAG system. Score:\n"
            "- faithfulness (1-5): are ALL claims in the answer supported by the context? "
            "Penalize anything not in the context.\n"
            "- correctness (1-5): does the answer agree with the reference answer?\n"
            "Give brief reasoning.",
        ),
        (
            "human",
            "Question: {question}\n\nReference answer: {reference}\n\n"
            "Retrieved context:\n{context}\n\nSystem answer: {answer}",
        ),
    ]
)


async def eval_one(chain, judge, row: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        t0 = time.perf_counter()
        out = await chain.ainvoke(
            {"question": row["question"], "history": []},
            config={"run_name": "eval_rag", "tags": ["eval"]},
        )
        latency = time.perf_counter() - t0
        sources = {d.metadata.get("source") for d in out["docs"]}
        hit = bool(sources & set(row.get("expected_sources", [])))
        j: Judgement = await judge.ainvoke(
            {
                "question": row["question"],
                "reference": row["reference_answer"],
                "context": format_docs(out["docs"]),
                "answer": out["answer"],
            }
        )
        return {
            "question": row["question"],
            "answer": out["answer"],
            "hit": hit,
            "faithfulness": j.faithfulness,
            "correctness": j.correctness,
            "reasoning": j.reasoning,
            "latency_s": round(latency, 2),
        }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/golden.jsonl")
    ap.add_argument("--out", default="eval/results.json")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--min-hit", type=float, default=0.8)
    ap.add_argument("--min-faith", type=float, default=4.0)
    ap.add_argument("--min-correct", type=float, default=3.5)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.golden).read_text().splitlines() if l.strip()]
    llm = get_llm()
    chain = build_chain(llm, HybridRetriever(get_vectorstore()))
    judge = JUDGE_PROMPT | llm.with_structured_output(Judgement)

    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*(eval_one(chain, judge, r, sem) for r in rows))

    summary = {
        "n": len(results),
        "hit_rate": sum(r["hit"] for r in results) / len(results),
        "faithfulness": statistics.mean(r["faithfulness"] for r in results),
        "correctness": statistics.mean(r["correctness"] for r in results),
        "p50_latency_s": statistics.median(r["latency_s"] for r in results),
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(json.dumps(summary, indent=2))

    ok = (
        summary["hit_rate"] >= args.min_hit
        and summary["faithfulness"] >= args.min_faith
        and summary["correctness"] >= args.min_correct
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
