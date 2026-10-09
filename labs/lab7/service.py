#!/usr/bin/env python3
"""Lab 7 HTTP service.

Run with ``uvicorn labs.lab7.service:app --port 8000``.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import sys
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip import cache, tracing  # noqa: E402
from aip.cache import CacheMiss  # noqa: E402
from aip.config import resolve_model, settings  # noqa: E402
from aip.cost import Budget, BudgetExceeded, global_budget  # noqa: E402
from aip.embed import cosine, embed  # noqa: E402
from labs.lab7.pipeline import answer_with_refusal_recovery, build_lab7_retriever  # noqa: E402


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    await asyncio.to_thread(pipeline)
    yield


app = FastAPI(title="Aurora Policy Assistant", version="1.0", lifespan=_lifespan)
_STARTED = time.time()
_PIPELINE: ServicePipeline | None = None
_RESPONSE_CACHE: dict[str, dict[str, Any]] = {}
_SEMANTIC_CACHE: list[tuple[str, list[float], dict[str, Any]]] = []
_CACHE_LOCK = RLock()
_SEMANTIC_THRESHOLD = 0.95
_SEMANTIC_MAX_ENTRIES = 512
_LOGGER = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="rag", pattern="^(rag|tools)$")


class Citation(BaseModel):
    index: int
    doc_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    sources: list[str]
    latency_ms: float
    cost_usd: float
    cached: bool
    trace_id: str


@dataclass
class ServicePipeline:
    """Process-lifetime integration of Labs 3-6."""

    retriever: Any

    def answer(self, req: AskRequest, trace_id: str) -> tuple[str, bool, list[Citation], list[str], float]:
        with Budget(limit_usd=0.01, label=f"lab7-request-{trace_id}") as budget:
            if req.mode == "tools":
                from labs.lab6 import agent

                guard = agent.make_guard(
                    read_only=True,
                    max_calls=6,
                    confirm_fn=lambda *_: False,
                )
                agent.set_defense_layers({1, 2, 3, 4, 5})
                with tracing.trace("lab7.tools", max_calls=guard.max_calls) as span:
                    result = agent.run_agent(
                        req.question,
                        guard=guard,
                        max_seconds=30.0,
                        budget_usd=0.01,
                    )
                    span["stopped_because"] = result.get("stopped_because")
                    span["tool_calls"] = len(result.get("tool_log", []))
                    span["refused"] = result.get("stopped_because") != "answered"
                if result.get("stopped_because") in {"spend_budget", "spend_preflight"}:
                    raise BudgetExceeded("Lab 6 agent exhausted the request budget")
                answer = str(result.get("answer", ""))
                used_tools = [
                    str(row.get("tool"))
                    for row in result.get("tool_log", [])
                    if row.get("ok")
                ]
                return answer, not bool(used_tools), [], used_tools, budget.spent_usd

            with tracing.trace("rag.retrieve", k=12) as retrieve_span:
                hits = self.retriever.search(req.question, k=12)
                retrieve_span["n_hits"] = len(hits)
                retrieve_span["top_doc"] = hits[0].doc_id if hits else None

            with tracing.trace("rag.generate_validate", top_k=req.top_k):
                result = answer_with_refusal_recovery(
                    req.question,
                    self.retriever,
                    k=12,
                    final_k=req.top_k,
                )

            citations = [
                Citation(index=index, doc_id=hit.doc_id, excerpt=hit.text[:1600])
                for index, hit in enumerate(result.hits, start=1)
                if index in _citation_numbers(result.text)
            ]
            source_ids = list(dict.fromkeys(hit.doc_id for hit in result.hits))
            return (
                result.text,
                result.refused,
                citations,
                source_ids,
                budget.spent_usd,
            )


def _citation_numbers(text: str) -> set[int]:
    return {int(value) for value in re.findall(r"\[(\d+)\]", text)}


def pipeline() -> ServicePipeline:
    """Build the Lab 3-5 RAG pipeline once and retain its index."""
    global _PIPELINE
    if _PIPELINE is None:
        with tracing.trace("lab7.pipeline.build", model=resolve_model("EMBED")):
            _PIPELINE = ServicePipeline(retriever=build_lab7_retriever())
    return _PIPELINE


def _normalised_question(req: AskRequest) -> str:
    text = " ".join(req.question.casefold().split())
    return f"{req.mode}|{req.top_k}|{text}"


def _cache_key(req: AskRequest) -> str:
    return hashlib.sha256(_normalised_question(req).encode("utf-8")).hexdigest()


def _cached_response(data: dict[str, Any], trace_id: str, elapsed_ms: float) -> AskResponse:
    response = AskResponse.model_validate(data)
    response.cached = True
    response.latency_ms = round(elapsed_ms, 2)
    response.cost_usd = 0.0
    response.trace_id = trace_id
    return response


def _semantic_entry(req: AskRequest, vector: list[float]) -> dict[str, Any] | None:
    scope = f"{req.mode}|{req.top_k}"
    question_key = _normalised_question(req)
    with _CACHE_LOCK:
        candidates = [
            (
                float(
                    cosine(
                        np.asarray(embed_vector, dtype=np.float32)[None, :],
                        np.asarray(vector, dtype=np.float32)[None, :],
                    )[0, 0]
                ),
                data,
            )
            for cached_scope, embed_vector, data in _SEMANTIC_CACHE
            if cached_scope.startswith(scope + "|")
            and cached_scope != f"{scope}|{question_key}"
        ]
    best_score, best_data = max(candidates, default=(-1.0, None), key=lambda item: item[0])
    tracing.event(
        "cache.semantic_lookup",
        similarity=round(best_score, 6) if best_score >= 0 else None,
        threshold=_SEMANTIC_THRESHOLD,
        hit=best_data is not None and best_score >= _SEMANTIC_THRESHOLD,
    )
    return best_data if best_data is not None and best_score >= _SEMANTIC_THRESHOLD else None


def _remember(req: AskRequest, response: AskResponse, vector: list[float] | None) -> None:
    data = response.model_copy(update={"cached": False}).model_dump()
    with _CACHE_LOCK:
        _RESPONSE_CACHE[_cache_key(req)] = data
        if vector is not None:
            _SEMANTIC_CACHE.append(
                (
                    f"{req.mode}|{req.top_k}|{_normalised_question(req)}",
                    vector,
                    data,
                )
            )
            if len(_SEMANTIC_CACHE) > _SEMANTIC_MAX_ENTRIES:
                del _SEMANTIC_CACHE[:-_SEMANTIC_MAX_ENTRIES]


def _is_provider_outage(exc: Exception) -> bool:
    module = type(exc).__module__.lower()
    name = type(exc).__name__.lower()
    status = getattr(exc, "status_code", None)
    provider_exception_names = {
        "apiconnectionerror",
        "apistatuserror",
        "internalservererror",
        "ratelimiterror",
        "serviceunavailableerror",
        "timeout",
        "timeouterror",
    }
    recognized_provider = (
        (module.startswith("litellm") or module.startswith("openai"))
        and name in provider_exception_names
    )
    response_status_error = (
        status in {408, 429, 500, 502, 503, 504}
        and getattr(exc, "response", None) is not None
    )
    return recognized_provider or response_status_error


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, BudgetExceeded):
        return HTTPException(status_code=429, detail="request budget exhausted")
    if isinstance(exc, CacheMiss):
        return HTTPException(
            status_code=503,
            detail="offline response cache miss",
            headers={"Retry-After": "10"},
        )
    if _is_provider_outage(exc):
        return HTTPException(
            status_code=503,
            detail="upstream model unavailable",
            headers={"Retry-After": "10"},
        )
    return HTTPException(status_code=500, detail="internal service error")


def _execute(req: AskRequest) -> AskResponse:
    started = time.perf_counter()
    trace_id = uuid4().hex
    try:
        return _execute_traced(req, trace_id, started)
    except Exception as exc:
        status_code = exc.status_code if isinstance(exc, HTTPException) else _http_error(exc).status_code
        tracing.event(
            "service.error",
            trace_id=trace_id,
            error_type=type(exc).__name__,
            error_kind=type(exc).__name__,
            status_code=status_code,
        )
        raise


def _execute_traced(req: AskRequest, trace_id: str, started: float) -> AskResponse:
    key = _cache_key(req)
    with tracing.trace("http.ask", trace_id=trace_id, question=req.question[:120], mode=req.mode) as span:
        with _CACHE_LOCK:
            exact = _RESPONSE_CACHE.get(key)
        if exact is not None:
            response = _cached_response(exact, trace_id, (time.perf_counter() - started) * 1000)
            span.update(cached=True, refused=response.refused, cost_usd=0.0)
            tracing.event("service.request", trace_id=trace_id, cached=True, cost_usd=0.0,
                          refused=response.refused, mode=req.mode)
            return response

        vector: list[float] | None = None
        try:
            with tracing.trace("cache.semantic_embed"):
                vector = embed(req.question, input_type="query").tolist()
            semantic = _semantic_entry(req, vector)
        except CacheMiss:
            semantic = None
            vector = None
            tracing.event("cache.semantic_lookup", hit=False, reason="embedding_cache_miss")

        if semantic is not None:
            response = _cached_response(
                semantic, trace_id, (time.perf_counter() - started) * 1000
            )
            span.update(cached=True, cache_layer="semantic", refused=response.refused,
                        cost_usd=0.0)
            tracing.event("service.request", trace_id=trace_id, cached=True,
                          cache_layer="semantic", cost_usd=0.0,
                          refused=response.refused, mode=req.mode)
            return response

        answer, refused, citations, sources, cost_usd = pipeline().answer(req, trace_id)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response = AskResponse(
            answer=answer,
            refused=refused,
            citations=citations,
            sources=sources,
            latency_ms=round(elapsed_ms, 2),
            cost_usd=round(cost_usd, 8),
            cached=False,
            trace_id=trace_id,
        )
        _remember(req, response, vector)
        span.update(cached=False, refused=refused, cost_usd=response.cost_usd)
        tracing.event("service.request", trace_id=trace_id, cached=False,
                      cost_usd=response.cost_usd, refused=refused, mode=req.mode)
        return response


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        return _execute(req)
    except HTTPException:
        raise
    except Exception as exc:
        error = _http_error(exc)
        raise error from exc


@app.get("/health")
def health() -> dict[str, Any]:
    index_size = 0
    if _PIPELINE is not None:
        index_size = len(getattr(_PIPELINE.retriever, "chunks", []))
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _STARTED, 1),
        "index_size": index_size,
        "model": resolve_model("MAIN"),
        "embedding_model": resolve_model("EMBED"),
        "cache": cache.stats(),
        "response_cache": {"exact_entries": len(_RESPONSE_CACHE),
                           "semantic_entries": len(_SEMANTIC_CACHE),
                           "semantic_threshold": _SEMANTIC_THRESHOLD},
    }


def _all_traces() -> list[dict[str, Any]]:
    records = []
    trace_dir = settings.trace_dir
    if not trace_dir.exists():
        return records
    for path in trace_dir.glob("*.jsonl"):
        try:
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        _LOGGER.warning("Skipping invalid trace %s:%d: %s", path, line_number, exc)
        except OSError as exc:
            _LOGGER.warning("Could not read trace file %s: %s", path, exc)
    return records


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    values.sort()
    index = max(0, math.ceil(percentile / 100 * len(values)) - 1)
    index = min(len(values) - 1, index)
    return round(values[index], 2)


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    rows = _all_traces()
    today_start = time.strptime(time.strftime("%Y-%m-%d"), "%Y-%m-%d")
    today_epoch = time.mktime(today_start)
    requests = [r for r in rows if r.get("name") == "http.ask"]
    today_requests = [r for r in requests if float(r.get("ts", 0)) >= today_epoch]
    today_trace_ids = {
        str(r.get("trace_id"))
        for r in today_requests
        if r.get("trace_id") is not None
    }
    successful_requests = [
        r for r in rows
        if r.get("name") == "service.request"
        and float(r.get("ts", 0)) >= today_epoch
    ]
    today_successful = [
        r for r in successful_requests
        if str(r.get("trace_id")) in today_trace_ids
    ]
    cost_today = sum(float(r.get("cost_usd", 0) or 0) for r in today_successful)
    latencies = [
        float(r.get("duration_ms", 0))
        for r in rows
        if r.get("name") == "http.ask"
        and float(r.get("ts", 0)) >= today_epoch
        and r.get("duration_ms") is not None
    ]
    errors: dict[str, int] = {}
    for row in rows:
        if (
            row.get("name") == "service.error"
            and float(row.get("ts", 0)) >= today_epoch
            and str(row.get("trace_id")) in today_trace_ids
        ):
            kind = str(row.get("error_kind") or row.get("error_type") or "unknown")
            errors[kind] = errors.get(kind, 0) + 1

    spans = {str(row.get("span_id")): row for row in rows if row.get("span_id")}
    today_http_spans = {
        str(row.get("span_id")) for row in today_requests if row.get("span_id")
    }
    lab7_tool_spans = {
        span_id for span_id, row in spans.items() if row.get("name") == "lab7.tools"
    }

    def has_ancestor(row: dict[str, Any], ancestors: set[str]) -> bool:
        parent_id = row.get("parent_id")
        visited: set[str] = set()
        while parent_id and str(parent_id) not in visited:
            parent = str(parent_id)
            if parent in ancestors:
                return True
            visited.add(parent)
            parent_id = spans.get(parent, {}).get("parent_id")
        return False

    calls: dict[str, int] = {}
    for row in rows:
        if (
            row.get("name") in {"tool.call", "tool.denied"}
            and float(row.get("ts", 0)) >= today_epoch
            and has_ancestor(row, lab7_tool_spans)
        ):
            tool = str(row.get("tool", "unknown"))
            calls[tool] = calls.get(tool, 0) + 1
    llm_calls = [
        row for row in rows
        if row.get("name") == "llm.call"
        and float(row.get("ts", 0)) >= today_epoch
        and has_ancestor(row, today_http_spans)
    ]
    if today_successful:
        cache_hits = sum(bool(r.get("cached")) for r in today_successful)
        cache_rate = cache_hits / len(today_successful)
    else:
        cache_hits = sum(bool(r.get("cached")) for r in llm_calls)
        cache_rate = cache_hits / len(llm_calls) if llm_calls else 0.0
    return {
        "cost_today_usd": round(cost_today, 6),
        "cost_per_query_usd": round(cost_today / len(today_requests), 8)
        if today_requests else 0.0,
        "queries_today": len(today_requests),
        "successful_queries_today": len(today_successful),
        "cache_hit_rate": round(cache_rate, 4),
        "latency_p50_ms": _percentile(latencies.copy(), 50),
        "latency_p95_ms": _percentile(latencies.copy(), 95),
        "latency_p99_ms": _percentile(latencies.copy(), 99),
        "error_rate_by_type": {
            kind: count / max(1, len(requests)) for kind, count in errors.items()
        },
        "error_counts_by_type": errors,
        "tool_call_counts": calls,
        "llm_calls": len(llm_calls),
        "process_budget": global_budget().as_dict(),
        "trace_count": len(rows),
    }


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """Validated-buffer SSE: emit prose only after final answer validation."""
    stream_started = time.perf_counter()
    try:
        response = await asyncio.to_thread(_execute, req)
    except HTTPException:
        raise
    except Exception as exc:
        error = _http_error(exc)
        raise error from exc

    async def events():
        yield {"event": "answer_start", "data": json.dumps({"trace_id": response.trace_id})}
        answer = response.answer
        for offset in range(0, len(answer), 48):
            ttft_ms = (time.perf_counter() - stream_started) * 1000
            yield {
                "event": "token",
                "data": json.dumps(
                    {
                        "text": answer[offset:offset + 48],
                        "ttft_ms": round(ttft_ms, 2) if offset == 0 else None,
                    },
                    ensure_ascii=False,
                ),
            }
        yield {
            "event": "citations",
            "data": json.dumps([c.model_dump() for c in response.citations], ensure_ascii=False),
        }
        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "trace_id": response.trace_id,
                    "latency_ms": response.latency_ms,
                    "cost_usd": response.cost_usd,
                    "cached": response.cached,
                    "ttft_ms": round((time.perf_counter() - stream_started) * 1000, 2),
                    "total_ms": round((time.perf_counter() - stream_started) * 1000, 2),
                }
            ),
        }

    return EventSourceResponse(events())
