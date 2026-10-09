"""Lab 7 retrieval recovery and Lab 3-5 RAG integration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aip import tracing
from aip.chunking import Chunk
from aip.retrieval import Bm25Retriever, Hit
from labs.lab3.search import build_chunks, load_corpus
from labs.lab4.evaluate import build_retriever as build_dense_retriever
from labs.lab4.rag import Answer, answer_question


@dataclass
class Lab7Retriever:
    """Keep the Lab 4 dense ranking and add lexical rescue for false refusals."""

    dense: Any
    lexical: Bm25Retriever
    chunks_by_doc: dict[str, list[Chunk]]

    def search(self, question: str, k: int = 8) -> list[Hit]:
        return self.dense.search(question, k=k)

    def search_after_refusal(self, question: str) -> list[Hit]:
        hits = self.lexical.search(question, k=1)
        if not hits:
            return []

        best = hits[0]
        selected = [best]
        siblings = self.chunks_by_doc[best.doc_id]
        position = next(
            (
                index for index, chunk in enumerate(siblings)
                if chunk.chunk_id == best.chunk.chunk_id
            ),
            None,
        )
        if len(best.text) <= 250 and position is not None and position + 1 < len(siblings):
            neighbor = siblings[position + 1]
            selected.append(
                Hit(
                    chunk=neighbor,
                    score=best.score,
                    source="bm25-neighbor",
                    rank=1,
                )
            )
        return selected


def build_lab7_retriever() -> Lab7Retriever:
    corpus = load_corpus()
    chunks = build_chunks(corpus, strategy="markdown", size=800)
    chunks_by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        chunks_by_doc.setdefault(chunk.doc_id, []).append(chunk)
    return Lab7Retriever(
        dense=build_dense_retriever(),
        lexical=Bm25Retriever(chunks),
        chunks_by_doc=chunks_by_doc,
    )


def answer_with_refusal_recovery(
    question: str,
    retriever: Lab7Retriever,
    *,
    k: int = 12,
    final_k: int = 5,
    reranker: Any = None,
    tier: str = "MAIN",
) -> Answer:
    """Retry an otherwise-safe refusal with the strongest lexical evidence.

    The retry uses the same Lab 4 citation validation and refusal fallback.
    If the focused evidence still cannot support an answer, retain the original
    refusal instead of weakening the response contract.
    """
    result = answer_question(
        question, retriever, k=k, final_k=final_k, reranker=reranker, tier=tier
    )
    if not result.refused:
        return result

    rescue_hits = retriever.search_after_refusal(question)
    if not rescue_hits:
        return result

    class FocusedEvidence:
        def search(self, _question: str, k: int = 8) -> list[Hit]:
            return rescue_hits[:k]

    with tracing.trace(
        "rag.refusal_recovery",
        source="bm25",
        n_hits=len(rescue_hits),
        original_sources=[hit.doc_id for hit in result.hits],
    ):
        recovered = answer_question(
            question,
            FocusedEvidence(),
            k=k,
            final_k=max(final_k, len(rescue_hits)),
            tier=tier,
        )
    return recovered if not recovered.refused else result
