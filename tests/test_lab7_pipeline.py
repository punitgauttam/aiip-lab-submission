from aip.retrieval import Chunk, Hit
from labs.lab4.rag import REFUSAL, Answer
from labs.lab7 import pipeline


def test_short_identifier_chunk_expands_to_next_section():
    first = Chunk("AUR-HI-SIL-2026", "plan-silver", "plan-silver::m0", {})
    next_section = Chunk(
        "Sum insured: ₹5,00,000 or ₹10,00,000.",
        "plan-silver",
        "plan-silver::m1",
        {},
    )

    class Lexical:
        def search(self, _question, k=1):
            return [Hit(first, 12.0, "bm25", 0)][:k]

    retriever = pipeline.Lab7Retriever(
        dense=None,
        lexical=Lexical(),
        chunks_by_doc={"plan-silver": [first, next_section]},
    )

    hits = retriever.search_after_refusal("AUR-HI-SIL-2026 sum insured options")

    assert [hit.chunk.chunk_id for hit in hits] == [
        "plan-silver::m0",
        "plan-silver::m1",
    ]
    assert hits[1].source == "bm25-neighbor"


def test_refusal_recovery_reuses_lab4_validation(monkeypatch):
    first = Answer(
        question="policy question",
        text=REFUSAL,
        refused=True,
    )
    recovered = Answer(
        question="policy question",
        text="Evidence-backed answer [1]",
        refused=False,
        citations_valid=True,
    )

    class Retriever:
        def search(self, _question, k=8):
            return []

        def search_after_refusal(self, _question):
            return [Hit(Chunk("evidence", "policy", "policy::m0", {}), 1.0)]

    calls = []

    def answer_question(_question, _retriever, **_kwargs):
        calls.append(_retriever)
        return first if len(calls) == 1 else recovered

    monkeypatch.setattr(pipeline, "answer_question", answer_question)
    result = pipeline.answer_with_refusal_recovery("policy question", Retriever())

    assert result is recovered
    assert len(calls) == 2
