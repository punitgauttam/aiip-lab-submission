#!/usr/bin/env python3
"""Offline-capable golden-set regression gate for Lab 7."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.cost import Budget  # noqa: E402
from aip.evals import retrieval_metrics  # noqa: E402
from aip.retrieval import format_context  # noqa: E402
from labs.lab3.search import load_questions  # noqa: E402
from labs.lab4.evaluate import judge_correctness, judge_faithfulness  # noqa: E402
from labs.lab7.pipeline import answer_with_refusal_recovery, build_lab7_retriever  # noqa: E402

LAST_ROWS: list[dict[str, Any]] = []


def measure() -> dict[str, float]:
    """Run the Lab 4/5 answer pipeline and judges on the committed golden set."""
    global LAST_ROWS
    questions = load_questions(include_unanswerable=True)
    retriever = build_lab7_retriever()
    rows = []
    latencies: list[float] = []
    costs: list[float] = []

    for question in questions:
        started = time.perf_counter()
        with Budget(limit_usd=1.0, label=f"lab7-gate-{question['id']}") as query_budget:
            answer = answer_with_refusal_recovery(question["question"], retriever)
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        costs.append(query_budget.spent_usd)

        context = format_context(answer.hits)
        correctness = judge_correctness(
            question["question"], answer.text, question["gold_answer"]
        ) / 2
        faithfulness = judge_faithfulness(answer.text, context)
        unanswerable = not question["relevant_docs"] or question["kind"] == "unanswerable"
        retrieved_docs = list(dict.fromkeys(hit.doc_id for hit in answer.hits))
        hit_rate = retrieval_metrics(
            retrieved_docs, question["relevant_docs"], ks=(5,)
        )["hit_rate@5"]
        rows.append(
            {
                "id": question["id"],
                "kind": question["kind"],
                "answer": answer.text,
                "correctness": correctness,
                "faithfulness": float(faithfulness),
                "citation_validity": float(answer.citations_valid),
                "refused": bool(answer.refused),
                "unanswerable": unanswerable,
                "hit_rate_at_5": hit_rate,
                "cost_usd": query_budget.spent_usd,
                "latency_ms": latency_ms,
            }
        )

    answerable = [row for row in rows if not row["unanswerable"]]
    unanswerable_rows = [row for row in rows if row["unanswerable"]]
    refusals = [row for row in rows if row["refused"]]
    refusal_recall = (
        sum(row["refused"] for row in unanswerable_rows) / len(unanswerable_rows)
        if unanswerable_rows else 0.0
    )
    refusal_precision = (
        sum(row["unanswerable"] for row in refusals) / len(refusals)
        if refusals else 1.0
    )
    LAST_ROWS = rows
    return {
        "correctness": statistics.fmean(row["correctness"] for row in answerable),
        "faithfulness": statistics.fmean(row["faithfulness"] for row in rows),
        "citation_validity": statistics.fmean(row["citation_validity"] for row in rows),
        "refusal_recall": refusal_recall,
        "refusal_precision": refusal_precision,
        "hit_rate_at_5": statistics.fmean(
            row["hit_rate_at_5"] for row in answerable
        ),
        "cost_per_query_usd": statistics.fmean(costs),
        "p95_latency_ms": _percentile(latencies, 95),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile / 100 * len(ordered)) - 1)
    return ordered[min(len(ordered) - 1, index)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="labs/lab7/thresholds.yml")
    parser.add_argument("--save", default="", help="save measured metrics and per-case rows as JSON")
    args = parser.parse_args()

    thresholds = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    metrics = measure()
    if args.save:
        output = ROOT / args.save
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"metrics": metrics, "rows": LAST_ROWS}, indent=2),
            encoding="utf-8",
        )
        print(f"saved evaluation -> {output}")

    failures = []
    width = max(len(name) for name in thresholds)
    print(f"{'metric':<{width}}  {'value':>10}  {'gate':>14}  status")
    print("-" * (width + 40))
    for name, rule in thresholds.items():
        value = metrics.get(name)
        if value is None:
            failures.append(f"{name}: not measured")
            print(f"{name:<{width}}  {'—':>10}  {'':>14}  MISSING")
            continue
        gate = ""
        ok = True
        if "min" in rule:
            gate = f">= {rule['min']}"
            ok = value >= rule["min"]
        if "max" in rule:
            gate = f"<= {rule['max']}"
            ok = ok and value <= rule["max"]
        if not ok:
            failures.append(f"{name}: {value} violates {gate}")
        print(f"{name:<{width}}  {value:>10.4f}  {gate:>14}  {'ok' if ok else 'FAIL'}")

    if failures:
        print("\nGATE FAILED:")
        for failure in failures:
            print("  " + failure)
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
