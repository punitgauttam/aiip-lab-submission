import time

from fastapi.testclient import TestClient

from labs.lab7 import service


def _response(**updates):
    payload = {
        "answer": "Covered by the policy. [1]",
        "refused": False,
        "citations": [],
        "sources": [],
        "latency_ms": 1.0,
        "cost_usd": 0.0,
        "cached": False,
        "trace_id": "test-trace",
    }
    payload.update(updates)
    return service.AskResponse(**payload)


def test_ask_rejects_invalid_request():
    response = TestClient(service.app).post("/ask", json={"question": "x"})
    assert response.status_code == 422


def test_ask_response_uses_documented_schema(monkeypatch):
    monkeypatch.setattr(service, "_execute", lambda _request: _response())
    response = TestClient(service.app).post("/ask", json={"question": "policy question"})
    assert response.status_code == 200
    assert set(response.json()) == {
        "answer",
        "refused",
        "citations",
        "sources",
        "latency_ms",
        "cost_usd",
        "cached",
        "trace_id",
    }


def test_budget_error_maps_to_429_without_internal_details(monkeypatch):
    monkeypatch.setattr(
        service,
        "_execute",
        lambda _request: (_ for _ in ()).throw(service.BudgetExceeded("private detail")),
    )
    response = TestClient(service.app).post("/ask", json={"question": "policy question"})
    assert response.status_code == 429
    assert "private detail" not in response.text


def test_provider_outage_maps_to_503_with_retry_after(monkeypatch):
    class ProviderError(Exception):
        status_code = 503
        response = object()

    monkeypatch.setattr(
        service,
        "_execute",
        lambda _request: (_ for _ in ()).throw(ProviderError("private detail")),
    )
    response = TestClient(service.app).post("/ask", json={"question": "policy question"})
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "10"
    assert "private detail" not in response.text


def test_programming_error_is_not_mislabeled_as_provider_outage(monkeypatch):
    monkeypatch.setattr(
        service,
        "_execute",
        lambda _request: (_ for _ in ()).throw(ValueError("private detail")),
    )
    response = TestClient(service.app).post("/ask", json={"question": "policy question"})
    assert response.status_code == 500
    assert "Retry-After" not in response.headers
    assert "private detail" not in response.text


def test_stream_emits_validated_citations_and_measures_ttft(monkeypatch):
    monkeypatch.setattr(
        service,
        "_execute",
        lambda _request: _response(
            citations=[service.Citation(index=1, doc_id="policy", excerpt="validated source")]
        ),
    )
    response = TestClient(service.app).post(
        "/ask/stream", json={"question": "policy question"}
    )
    assert response.status_code == 200
    assert "event: token" in response.text
    assert "event: citations" in response.text
    assert "validated source" in response.text
    assert "ttft_ms" in response.text
    assert "total_ms" in response.text


def test_semantic_cache_only_matches_same_mode_and_top_k(monkeypatch):
    req = service.AskRequest(question="What is my policy coverage?", mode="rag", top_k=5)
    response = _response()
    vector = [1.0, 0.0]
    service._SEMANTIC_CACHE.clear()
    service._remember(req, response, vector)

    related = service.AskRequest(
        question="Tell me about my policy coverage", mode="rag", top_k=5
    )
    monkeypatch.setattr(service, "_SEMANTIC_THRESHOLD", 0.95)
    assert service._semantic_entry(related, vector) is not None

    different_mode = related.model_copy(update={"mode": "tools"})
    different_top_k = related.model_copy(update={"top_k": 3})
    assert service._semantic_entry(different_mode, vector) is None
    assert service._semantic_entry(different_top_k, vector) is None
    service._SEMANTIC_CACHE.clear()


def test_metrics_are_scoped_to_lab7_requests(monkeypatch):
    now = time.time()
    traces = [
        {
            "name": "http.ask",
            "trace_id": "success",
            "ts": now,
            "duration_ms": 10.0,
            "status": "ok",
        },
        {
            "name": "http.ask",
            "trace_id": "failure",
            "ts": now,
            "duration_ms": 20.0,
            "status": "error",
        },
        {
            "name": "service.request",
            "trace_id": "success",
            "ts": now,
            "cached": True,
            "cost_usd": 0.005,
        },
        {
            "name": "service.error",
            "trace_id": "failure",
            "ts": now,
            "error_type": "ProviderError",
        },
        {"name": "lab7.tools", "span_id": "tools-span", "ts": now},
        {"name": "tool.call", "parent_id": "tools-span", "tool": "search_policy", "ts": now},
        {"name": "tool.call", "parent_id": None, "tool": "issue_refund", "ts": now},
        {
            "name": "service.error",
            "ts": now - 86400,
            "error_type": "old-error",
        },
    ]
    monkeypatch.setattr(service, "_all_traces", lambda: traces)

    result = service.metrics()

    assert result["queries_today"] == 2
    assert result["cost_today_usd"] == 0.005
    assert result["cost_per_query_usd"] == 0.0025
    assert result["cache_hit_rate"] == 1.0
    assert result["error_counts_by_type"] == {"ProviderError": 1}
    assert result["error_rate_by_type"] == {"ProviderError": 0.5}
    assert result["tool_call_counts"] == {"search_policy": 1}
