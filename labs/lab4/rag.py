#!/usr/bin/env python3
"""Lab 4 — grounded RAG answer pipeline."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.guards import UNTRUSTED_SYSTEM_CLAUSE, delimit_untrusted
from aip.llm import chat
from aip.retrieval import Hit, Retriever, format_context


REFUSAL = "I don't have enough information in the provided sources to answer that."


# ---------------------------------------------------------------------------
# Generation policy
# ---------------------------------------------------------------------------
# The Lab 4 prompt is preserved verbatim for the Lab 5 baseline. The active
# Lab 5 prompt below changes only its answer-format instruction.
ANSWER_SYSTEM_BASELINE = f"""
You are a grounded question-answering system.

You MUST follow these rules:

1. SOURCE-ONLY ANSWERS
Answer using ONLY the supplied numbered sources.
Do not use general knowledge, memory, assumptions, or outside information.

2. EVIDENCE-FIRST REASONING
Before writing the final answer, determine what factual pieces are required
to answer the question.

For multi-part or multi-hop questions:
- break the question into its individual claims or sub-questions;
- identify which supplied source(s) support each required piece;
- combine information across sources only when the sources support the
  combination;
- do not fill a missing step with an assumption.

3. CITATIONS
Every factual claim in the final answer must have a citation.
Use the exact source numbers, for example:
[1]
[2][5]

Never invent a citation number.
Only cite source numbers that actually appear in the supplied context.

4. PARTIAL ANSWERS
If a question contains multiple parts and the sources support only some of
them:
- answer the supported part;
- clearly state that the missing part cannot be determined from the supplied
  sources;
- do NOT refuse the entire question merely because one part is unsupported.

If none of the requested information is supported by the sources, output
exactly:

{REFUSAL}

5. CONTRADICTORY SOURCES
If supplied sources disagree:
- report the disagreement;
- cite the relevant sources;
- do not silently choose one version.

6. NO UNSUPPORTED INFERENCE
A citation being relevant is not enough. The cited source must actually
support the claim being made.
Do not strengthen, extend, calculate, or infer information that the sources
do not establish.

7. ANSWER FORMAT
Normally answer in 2–3 sentences.
For a genuinely multi-part question, use as many short sentences as necessary
to cover the supported parts accurately.

Do not include your internal reasoning or analysis.
Return only the final answer with citations.

{UNTRUSTED_SYSTEM_CLAUSE}
"""

# Lab 5's single experimental change: focus on the requested facts and avoid
# adding adjacent details that can introduce errors. Keep the Lab 4 prompt
# intact so the before/after run can replay its cached baseline exactly.
ANSWER_SYSTEM = ANSWER_SYSTEM_BASELINE.replace(
    """7. ANSWER FORMAT
Normally answer in 2–3 sentences.
For a genuinely multi-part question, use as many short sentences as necessary
to cover the supported parts accurately.""",
    """7. ANSWER FORMAT
Answer the exact question directly. For a single-part question, use one short
sentence when that is sufficient. Do not add related thresholds, exceptions,
or product comparisons unless the question asks for them or omitting them would
make the direct answer misleading. For a multi-part question, cover each
requested part in a short sentence.""",
)
if ANSWER_SYSTEM == ANSWER_SYSTEM_BASELINE:
    raise RuntimeError("Lab 5 answer-format prompt was not applied")


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    refused: bool = False
    citations_valid: bool = False
    invalid_citations: list[int] = field(default_factory=list)
    n_citations: int = 0
    truncated: bool = False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_answer(
    text: str,
    n_sources: int,
    finish_reason: str | None = None,
) -> dict:
    """Validate the answer contract.

    A bad citation or truncated answer is never allowed to pass through.
    """

    text = (text or "").strip()

    citations = [int(x) for x in re.findall(r"\[(\d+)\]", text)]
    invalid = sorted({n for n in citations if n < 1 or n > n_sources})

    refused = text == REFUSAL
    truncated = finish_reason == "length"

    if not text:
        return {
            "valid": False,
            "refused": False,
            "invalid_citations": invalid,
            "n_citations": len(citations),
            "truncated": truncated,
            "reason": "empty_answer",
        }

    if truncated:
        return {
            "valid": False,
            "refused": refused,
            "invalid_citations": invalid,
            "n_citations": len(citations),
            "truncated": True,
            "reason": "truncated",
        }

    if invalid:
        return {
            "valid": False,
            "refused": refused,
            "invalid_citations": invalid,
            "n_citations": len(citations),
            "truncated": False,
            "reason": "invalid_citation",
        }

    if not refused and not citations:
        return {
            "valid": False,
            "refused": False,
            "invalid_citations": [],
            "n_citations": 0,
            "truncated": False,
            "reason": "missing_citation",
        }

    return {
        "valid": True,
        "refused": refused,
        "invalid_citations": [],
        "n_citations": len(citations),
        "truncated": False,
        "reason": "ok",
    }


# ---------------------------------------------------------------------------
# LLM response extraction
# ---------------------------------------------------------------------------
def _extract_chat_text(result):
    """Handle the toolkit's possible chat return shapes."""

    if isinstance(result, str):
        return result.strip(), None

    if hasattr(result, "text"):
        text = getattr(result, "text")
        finish = getattr(result, "finish_reason", None)
        return (text or "").strip(), finish

    if hasattr(result, "message"):
        message = result.message
        if hasattr(message, "content"):
            return (
                (message.content or "").strip(),
                getattr(result, "finish_reason", None),
            )

    if isinstance(result, dict):
        if "text" in result:
            return (
                (result.get("text") or "").strip(),
                result.get("finish_reason"),
            )

        message = result.get("message")
        if isinstance(message, dict):
            return (
                (message.get("content") or "").strip(),
                result.get("finish_reason"),
            )

    return str(result).strip(), None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _generate(question: str, context: str, tier: str):
    prompt = f"""
Question:
{question}

Numbered sources:
{context}

Instructions:
- First identify the distinct pieces of information needed to answer the
  question.
- Check each piece against the numbered sources.
- For multi-hop questions, combine only explicitly supported pieces.
- If one part is supported and another is not, answer the supported part and
  explicitly refuse only the missing part.
- If nothing relevant is supported, output exactly:
{REFUSAL}

Return only the final answer with citations.
"""

    result = chat(prompt, system=ANSWER_SYSTEM, tier=tier)
    return _extract_chat_text(result)


def _repair_answer(
    question: str,
    context: str,
    previous_answer: str,
    validation_reason: str,
    tier: str,
):
    """One corrective generation attempt.

    We retry instead of returning an answer that violates the contract.
    """

    prompt = f"""
The previous answer failed validation.

Question:
{question}

Numbered sources:
{context}

Previous answer:
{previous_answer}

Validation failure:
{validation_reason}

Rewrite the answer.

Requirements:
- Use ONLY the numbered sources.
- Every factual claim needs a valid citation.
- Never invent citation numbers.
- For multi-part questions, answer each supported part separately.
- If only part of the question is supported, give the supported part and
  explicitly state what cannot be determined from the sources.
- If nothing is supported, output exactly:
{REFUSAL}
- Do not include reasoning or commentary.
"""

    result = chat(prompt, system=ANSWER_SYSTEM, tier=tier)
    return _extract_chat_text(result)


# ---------------------------------------------------------------------------
# Main retrieval + generation pipeline
# ---------------------------------------------------------------------------
def answer_question(
    question: str,
    retriever: Retriever,
    *,
    k: int = 12,
    final_k: int = 5,
    reranker=None,
    tier: str = "MAIN",
) -> Answer:

    # 1. Retrieve
    hits = retriever.search(question, k=k)

    # 2. Optional reranking
    if reranker is not None:
        try:
            hits = reranker.rerank(question, hits)
        except Exception:
            # Do not let an optional reranker break the core pipeline.
            pass

    hits = list(hits[:final_k])
    context = format_context(hits)

    # 3. Generate
    text, finish_reason = _generate(question, context, tier)

    # 4. Validate
    check = validate_answer(text, len(hits), finish_reason)

    # 5. Repair once if validation fails
    if not check["valid"]:
        text, finish_reason = _repair_answer(
            question,
            context,
            text,
            check["reason"],
            tier,
        )
        check = validate_answer(text, len(hits), finish_reason)

    # 6. Safety fallback.
    # Never return an uncited / invalid answer as a successful answer.
    if not check["valid"]:
        return Answer(
            question=question,
            text=REFUSAL,
            hits=hits,
            refused=True,
            citations_valid=True,
            invalid_citations=[],
            n_citations=0,
            truncated=check["truncated"],
        )

    return Answer(
        question=question,
        text=text,
        hits=hits,
        refused=check["refused"],
        citations_valid=check["valid"],
        invalid_citations=check["invalid_citations"],
        n_citations=check["n_citations"],
        truncated=check["truncated"],
    )


# ---------------------------------------------------------------------------
# Gold-context generation for E2
# ---------------------------------------------------------------------------
def answer_with_gold_context(
    question: str,
    gold_docs: list[str],
    *,
    tier: str = "MAIN",
) -> Answer:
    """Same generator, but with gold documents and no retrieval."""

    context_parts = []

    for i, doc in enumerate(gold_docs, start=1):
        # Delimit document text as untrusted evidence.
        context_parts.append(
            f"[{i}]\n{delimit_untrusted(doc)}"
        )

    context = "\n\n".join(context_parts)

    text, finish_reason = _generate(question, context, tier)

    check = validate_answer(text, len(gold_docs), finish_reason)

    if not check["valid"]:
        text, finish_reason = _repair_answer(
            question,
            context,
            text,
            check["reason"],
            tier,
        )
        check = validate_answer(text, len(gold_docs), finish_reason)

    if not check["valid"]:
        return Answer(
            question=question,
            text=REFUSAL,
            hits=[],
            refused=True,
            citations_valid=True,
            invalid_citations=[],
            n_citations=0,
            truncated=check["truncated"],
        )

    return Answer(
        question=question,
        text=text,
        hits=[],
        refused=check["refused"],
        citations_valid=check["valid"],
        invalid_citations=check["invalid_citations"],
        n_citations=check["n_citations"],
        truncated=check["truncated"],
    )
