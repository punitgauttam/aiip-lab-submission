#!/usr/bin/env python3
"""Lab 3 — retrieval sweeps.

The scaffolding (corpus loading, metric computation, table printing) is
written for you. The sweeps are yours.

    python labs/lab3/search.py --baseline
    python labs/lab3/search.py --sweep chunking
    python labs/lab3/search.py --sweep retrieval
    python labs/lab3/search.py --sweep rerank
    python labs/lab3/search.py --sweep index
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import STRATEGIES, Chunk  # noqa: E402
from aip.cost import Budget  # noqa: E402
from aip.evals import retrieval_metrics  # noqa: E402
from aip.retrieval import (  # noqa: E402
    Bm25Retriever,
    ChromaRetriever,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
    LLMReranker,
    Retriever,
)

CORPUS_DIR = ROOT / "data/corpus"
GOLDEN = ROOT / "data/eval/rag_golden.jsonl"


# ---------------------------------------------------------------------------
# scaffolding (provided)
# ---------------------------------------------------------------------------
def load_corpus() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(CORPUS_DIR.glob("*.md"))}


def load_questions(include_unanswerable: bool = False) -> list[dict]:
    rows = [json.loads(l) for l in GOLDEN.open(encoding="utf-8")]
    if include_unanswerable:
        return rows
    # THREE questions (Q36, Q38, Q39) have no relevant document, so recall and
    # nDCG are undefined for them -- you cannot rank correctly against an empty
    # relevant set. Dropping them leaves n = 42.
    #
    # Do not confuse that with the FIVE questions of kind 'unanswerable'
    # (Q36-Q40): two of those do keep relevant documents, because part of what
    # they ask is supported. All five are measured properly in Lab 4, as
    # refusal precision and recall.
    #
    # Excluding the three is correct -- but say so in your report rather than
    # letting an unexplained n = 42 pass for a stated 45.
    return [r for r in rows if r["relevant_docs"]]


def build_chunks(corpus: dict[str, str], strategy: str = "sliding",
                 size: int = 800, **kw) -> list[Chunk]:
    fn = STRATEGIES[strategy]
    out: list[Chunk] = []
    for doc_id, text in corpus.items():
        try:
            out.extend(fn(text, doc_id, size=size, **kw))
        except TypeError:                       # chunker without that kwarg
            out.extend(fn(text, doc_id, size=size))
    return out


def evaluate(retriever: Retriever, questions: list[dict], k: int = 10,
             reranker=None, final_k: int = 5) -> dict:
    """Run every question, return aggregate metrics + per-kind breakdown."""
    agg: dict[str, list[float]] = defaultdict(list)
    by_kind: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    latencies: list[float] = []
    per_q: dict[str, float] = {}
    per_q_mrr: dict[str, float] = {}

    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], k=k)
        if reranker is not None:
            hits = reranker.rerank(q["question"], hits, k=final_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        # A document counts as retrieved at rank r if any of its chunks does.
        seen, ranked = set(), []
        for h in hits:
            if h.doc_id not in seen:
                seen.add(h.doc_id)
                ranked.append(h.doc_id)

        m = retrieval_metrics(ranked, q["relevant_docs"], ks=(1, 3, 5, 10))
        per_q[q["id"]] = m["hit_rate@5"]
        per_q_mrr[q["id"]] = m["mrr"]
        for key, val in m.items():
            agg[key].append(val)
            by_kind[q["kind"]][key].append(val)

    out = {k2: statistics.fmean(v) for k2, v in agg.items()}
    out["latency_p50_ms"] = statistics.median(latencies)
    out["latency_p95_ms"] = sorted(latencies)[int(0.95 * (len(latencies) - 1))]
    out["_by_kind"] = {kind: {k2: statistics.fmean(v) for k2, v in d.items()}
                       for kind, d in by_kind.items()}
    out["_per_question"] = per_q            # hit_rate@5 -- saturated, see kind_table
    out["_per_question_mrr"] = per_q_mrr    # use this one for Part B
    out["_kind_n"] = {kind: len(d["mrr"]) for kind, d in by_kind.items()}
    return out


def table(rows: dict[str, dict], cols: tuple[str, ...] =
          ("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10",
           "latency_p95_ms")) -> str:
    name_w = max(len(n) for n in rows) + 2
    head = f"{'config':<{name_w}}" + "".join(f"{c:>15}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, m in rows.items():
        lines.append(f"{name:<{name_w}}" + "".join(f"{m.get(c, 0):>15.4f}" for c in cols))
    return "\n".join(lines)


def kind_table(metrics: dict, col: str = "hit_rate@5") -> str:
    """Break a result down by question kind.

    NOTE the default column. `hit_rate@5` is saturated on this corpus -- every
    retriever scores 0.93-0.98 -- so this table will look flat and tell you
    nothing. Pass col='mrr' or col='ndcg@10' for Part B. The default is left
    saturated on purpose.
    """
    bk, counts = metrics["_by_kind"], metrics.get("_kind_n", {})
    w = max(len(k) for k in bk) + 2
    lines = [f"{'kind':<{w}}{col:>12}{'n':>6}", "-" * (w + 18)]
    for kind, m in sorted(bk.items()):
        lines.append(f"{kind:<{w}}{m.get(col, 0):>12.4f}{counts.get(kind, 0):>6}")
    return "\n".join(lines)


def strip_markdown_prefix(chunks: list[Chunk]) -> list[Chunk]:
    """Strip the markdown heading prefix prepended by markdown_chunks()."""
    out: list[Chunk] = []
    for chunk in chunks:
        text = chunk.text
        if text.startswith("[") and "]" in text:
            end = text.find("]\n")
            if end != -1:
                text = text[end + 2 :]
        out.append(Chunk(text, chunk.doc_id, chunk.chunk_id, dict(chunk.meta)))
    return out


def _run_metric_suite(name: str, retriever: Retriever, questions: list[dict], *,
                     k: int = 10, reranker=None, final_k: int = 5) -> tuple[dict, str]:
    m = evaluate(retriever, questions, k=k, reranker=reranker, final_k=final_k)
    return m, table({name: m})


def _load_corpus_dir(path: Path) -> dict[str, str]:
    """Load *.md files from an arbitrary directory, same shape as
    load_corpus(). Used for D2's filler/ballast documents, which
    scripts/expand_corpus.py writes to data/corpus_scaled/ -- a SEPARATE
    directory from data/corpus/, confirmed by a real run:
        'wrote 400 filler documents to ...\\data\\corpus_scaled'
    load_corpus() only ever reads data/corpus/, so without this, D2 silently
    re-measures the same unscaled corpus at every "scale" and chunk_count
    never moves.
    """
    if not path.exists():
        return {}
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(path.glob("*.md"))}


def _time_queries(retriever: Retriever, questions: list[dict],
                  repeats: int = 3, k: int = 10) -> tuple[float, float]:
    """Median and p95 query latency (ms), averaged over `repeats` calls per
    question. Used by D2's exact-vs-ANN scale sweep."""
    times: list[float] = []
    for q in questions:
        for _ in range(repeats):
            t0 = time.perf_counter()
            retriever.search(q["question"], k=k)
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p50 = statistics.median(times)
    p95 = times[int(0.95 * (len(times) - 1))]
    return p50, p95


def _extract_cost_usd(budget: Budget) -> float:
    """Best-effort extraction of total cost from a Budget context manager."""
    cost = getattr(budget, "cost_usd", None)
    if isinstance(cost, (int, float)):
        return float(cost)

    report = budget.report()
    if isinstance(report, dict):
        val = report.get("cost_usd", 0.0)
        if isinstance(val, (int, float)):
            return float(val)
    elif isinstance(report, str):
        m = re.search(r"cost=\$([\d.]+)", report)
        if m:
            return float(m.group(1))

    print("  !! could not extract a cost figure from the Budget report -- "
          "cost_per_1k below is unreliable, not necessarily zero.")
    return 0.0


# ---------------------------------------------------------------------------
# sweeps (yours)
# ---------------------------------------------------------------------------
def sweep_baseline() -> None:
    corpus, questions = load_corpus(), load_questions()
    chunks = build_chunks(corpus, "sliding", 800, overlap=150)
    print(f"corpus: {len(corpus)} docs -> {len(chunks)} chunks "
          f"(mean {statistics.fmean(len(c) for c in chunks):.0f} chars)")
    r = DenseRetriever(chunks)
    m = evaluate(r, questions)
    print(table({"baseline sliding-800 dense": m}))
    print()
    print(kind_table(m))
    print("\nWrite these numbers down before you change anything.")


def sweep_chunking() -> None:
    """Measure the biggest lever in retrieval: chunking strategy and size."""
    corpus = load_corpus()
    questions = load_questions()
    strategies = ["fixed", "sliding", "recursive", "markdown"]
    rows: dict[str, dict] = {}

    for strategy in strategies:
        start = time.perf_counter()
        chunks = build_chunks(corpus, strategy, 800)
        r = DenseRetriever(chunks, show_progress=False)
        metrics = evaluate(r, questions)
        rows[strategy] = {
            "chunk_count": len(chunks),
            "build_ms": round((time.perf_counter() - start) * 1000, 1),
            **{k: v for k, v in metrics.items() if not k.startswith("_")},
        }

    print("Chunking sweep at 800 chars")
    print(table(rows, cols=("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10", "latency_p95_ms")))
    print("\nChunk count / build time")
    print(table({name: {"chunk_count": info["chunk_count"], "build_ms": info["build_ms"]} for name, info in rows.items()}, cols=("chunk_count", "build_ms")))

    winner = max(strategies, key=lambda s: rows[s]["ndcg@10"])
    print(f"\nWinner by nDCG@10: {winner}")
    print("Size sweep for the winner")
    size_rows: dict[str, dict] = {}
    for size in (400, 800, 1600):
        start = time.perf_counter()
        chunks = build_chunks(corpus, winner, size)
        r = DenseRetriever(chunks, show_progress=False)
        metrics = evaluate(r, questions)
        size_rows[f"{winner}:{size}"] = {
            "chunk_count": len(chunks),
            "build_ms": round((time.perf_counter() - start) * 1000, 1),
            **{k: v for k, v in metrics.items() if not k.startswith("_")},
        }
    print(table(size_rows, cols=("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10", "latency_p95_ms")))

    print("\nMarkdown prefix check")
    md_chunks = build_chunks(corpus, "markdown", 800)
    md_with = DenseRetriever(md_chunks, show_progress=False)
    md_without = DenseRetriever(strip_markdown_prefix(md_chunks), show_progress=False)
    with_metrics = evaluate(md_with, questions)
    without_metrics = evaluate(md_without, questions)
    print(table({"markdown_prefix_on": with_metrics, "markdown_prefix_off": without_metrics}, cols=("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10")))

    # -----------------------------------------------------------------
    # A4 -- find one question where chunking is clearly the failure, and
    # print the chunk that SHOULD have matched next to the chunks that DID.
    # Uses the winning strategy at its winning size from the sweep above.
    # -----------------------------------------------------------------
    print("\nA4 -- a chunking failure, in the flesh")
    best_name = max(size_rows, key=lambda n: size_rows[n]["ndcg@10"])
    best_size = int(best_name.split(":")[1])
    best_chunks = build_chunks(corpus, winner, best_size)
    best_retriever = DenseRetriever(best_chunks, show_progress=False)
    best_metrics = evaluate(best_retriever, questions)
    per_q_mrr = best_metrics["_per_question_mrr"]
    worst_qid = min(per_q_mrr, key=per_q_mrr.get)
    worst_q = next(q for q in questions if q["id"] == worst_qid)

    print(f"Worst question on the winning config ({best_name}): "
          f"{worst_qid}  (mrr={per_q_mrr[worst_qid]:.3f})")
    print(f"  question: {worst_q['question']}")
    print(f"  relevant doc(s): {worst_q['relevant_docs']}")

    gold_chunks = [c for c in best_chunks if c.doc_id in worst_q["relevant_docs"]]
    print("\n  chunk(s) that SHOULD have matched:")
    for c in gold_chunks[:3]:
        print(f"    [{c.doc_id}] {c.text[:200]!r}")

    retrieved = best_retriever.search(worst_q["question"], k=5)
    print("\n  chunks that WERE retrieved instead:")
    for h in retrieved:
        print(f"    [{h.doc_id}] score={h.score:.3f} {h.text[:200]!r}")
    print("\n  -> put this in your report as the A4 example. If the gold chunk")
    print("     and the retrieved chunks look unrelated, this is failure mode 2")
    print("     from T4 §5 (the answer got split across a chunk boundary).")
    print("     If the worst question here is actually a bad fit for this")
    print("     diagnostic (e.g. it's genuinely unanswerable), pick the next")
    print("     worst id from best_metrics['_per_question_mrr'] instead.")


def sweep_retrieval() -> None:
    """Compare dense, BM25, and hybrid retrieval on the best chunking."""
    corpus = load_corpus()
    questions = load_questions()
    chunks = build_chunks(corpus, "markdown", 800)

    configs: dict[str, Retriever] = {
        "dense": DenseRetriever(chunks, show_progress=False),
        "bm25": Bm25Retriever(chunks),
        "hybrid_rrf_60": HybridRetriever([DenseRetriever(chunks, show_progress=False), Bm25Retriever(chunks)], rrf_k=60),
    }

    results: dict[str, dict] = {}
    for name, retriever in configs.items():
        results[name] = evaluate(retriever, questions)

    print("Retrieval comparison on markdown-800")
    print(table(results, cols=("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10", "latency_p95_ms")))
    for name, metrics in results.items():
        print(f"\n{name}\n{kind_table(metrics, col='mrr')}")

    q44 = next(q for q in questions if q["id"] == "Q44")
    q41 = next(q for q in questions if q["id"] == "Q41")
    print("\nPer-question MRR on the two key examples")
    for qid, label in (("Q44", q44), ("Q41", q41)):
        row = {}
        for name, retriever in configs.items():
            m = evaluate(retriever, [label])
            row[name] = m["mrr"]
        print(f"{qid}: {row}")

    print("\nRRF k sweep")
    for k in (10, 30, 60, 100):
        r = HybridRetriever([DenseRetriever(chunks, show_progress=False), Bm25Retriever(chunks)], rrf_k=k)
        metrics = evaluate(r, questions)
        print(f"rrf_k={k}: hit_rate@1={metrics['hit_rate@1']:.4f}, mrr={metrics['mrr']:.4f}, ndcg@10={metrics['ndcg@10']:.4f}")

    # -----------------------------------------------------------------
    # B4 -- try more than one weighting so "does anything beat 1:1" is an
    # actual comparison rather than a single anecdote. n=42 is small, so the
    # printed note below is not decorative -- treat small deltas as noise.
    # -----------------------------------------------------------------
    print("\nWeighted fusion sweep")
    weight_options = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (3.0, 1.0), (1.0, 3.0)]
    weighted_rows: dict[str, dict] = {}
    for w_dense, w_bm25 in weight_options:
        r = HybridRetriever(
            [DenseRetriever(chunks, show_progress=False), Bm25Retriever(chunks)],
            weights=[w_dense, w_bm25],
        )
        weighted_rows[f"dense:{w_dense}_bm25:{w_bm25}"] = evaluate(r, questions)
    print(table(weighted_rows, cols=("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10")))

    baseline_key = "dense:1.0_bm25:1.0"
    best_weighted = max(weighted_rows, key=lambda n: weighted_rows[n]["ndcg@10"])
    baseline_ndcg = weighted_rows[baseline_key]["ndcg@10"]
    best_ndcg = weighted_rows[best_weighted]["ndcg@10"]
    print(f"\nBest weighting: {best_weighted} (ndcg@10={best_ndcg:.4f}) "
          f"vs 1:1 baseline (ndcg@10={baseline_ndcg:.4f}), "
          f"delta={best_ndcg - baseline_ndcg:+.4f}")
    print("Note: at n=42, deltas under roughly 0.02-0.03 ndcg@10 are well")
    print("within noise -- don't report a 'winning' ratio without checking")
    print("that against a paired test, same as Lab 2's discipline.")


def sweep_rerank() -> None:
    """Measure cross-encoder and LLM reranking on top of a wide retrieval pass."""
    corpus = load_corpus()
    questions = load_questions()
    chunks = build_chunks(corpus, "markdown", 800)
    base = DenseRetriever(chunks, show_progress=False)
    base_eval = evaluate(base, questions, k=30)

    rr = CrossEncoderReranker()
    reranked_eval = evaluate(base, questions, k=30, reranker=rr, final_k=5)

    llm_rr = LLMReranker(tier="SMALL")
    with Budget(limit_usd=2.0, label="lab3-llm-rerank") as b:
        llm_eval = evaluate(base, questions, k=30, reranker=llm_rr, final_k=5)
    print("\nLLM reranker budget report:")
    print(b.report())
    llm_cost_usd = _extract_cost_usd(b)
    llm_cost_per_1k = (llm_cost_usd / max(len(questions), 1)) * 1000

    # C1/C2/C3 -- decision table needs nDCG@5 (not @10), hit_rate@1, p95
    # latency, and $/1k queries per the README. Local rerankers cost $0/query.
    base_eval["cost_per_1k"] = 0.0
    reranked_eval["cost_per_1k"] = 0.0
    llm_eval["cost_per_1k"] = llm_cost_per_1k

    print("\nReranking comparison")
    print(table({
        "dense_k30": base_eval,
        "dense_k30_crossencoder": reranked_eval,
        "dense_k30_llm": llm_eval,
    }, cols=("ndcg@5", "hit_rate@1", "recall@5", "latency_p95_ms", "cost_per_1k")))

    print("\nC3 -- decision inputs (write the actual recommendation in report.md)")
    print(f"  interactive search box needs low p95: dense_k30={base_eval['latency_p95_ms']:.0f} ms, "
          f"crossencoder={reranked_eval['latency_p95_ms']:.0f} ms, llm={llm_eval['latency_p95_ms']:.0f} ms")
    print(f"  overnight batch can absorb latency/cost for quality: "
          f"llm ndcg@5={llm_eval['ndcg@5']:.4f} at ${llm_eval['cost_per_1k']:.2f}/1k queries")
    print("  -> if the LLM reranker's ndcg@5 gain over dense_k30 doesn't clearly")
    print("     beat its added latency+cost for the interactive case, that's")
    print("     your two-different-answers argument for C3.")

    before = base_eval["_per_question_mrr"]
    after = reranked_eval["_per_question_mrr"]
    delta = {qid: after.get(qid, 0.0) - before.get(qid, 0.0) for qid in before}
    worst = min(delta, key=delta.get)
    print(f"\nWorst query after cross-encoder reranking: {worst} with delta={delta[worst]:.4f}")


def sweep_index() -> None:
    """Compare exact dense search to Chroma ANN and test metadata filtering."""
    corpus = load_corpus()
    questions = load_questions()
    chunks = build_chunks(corpus, "markdown", 800)
    for chunk in chunks:
        chunk.meta["status"] = "archived" if "ARCHIVED" in chunk.doc_id else "current"

    exact = DenseRetriever(chunks, show_progress=False)
    ann = ChromaRetriever(chunks, path=".chroma", collection="lab3_demo", reset=True)

    exact_metrics = evaluate(exact, questions)
    ann_metrics = evaluate(ann, questions)
    print("Exact vs Chroma comparison")
    print(table({"dense_exact": exact_metrics, "chroma_hnsw": ann_metrics}, cols=("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10", "latency_p95_ms")))

    print("\nBefore/after metadata filtering on Q29-Q31")
    target_ids = ["Q29", "Q30", "Q31"]
    qmap = {q["id"]: q for q in questions}
    for qid in target_ids:
        q = qmap[qid]
        before_hits = ann.search(q["question"], k=5)
        before_doc_ids = [h.doc_id for h in before_hits]
        before = retrieval_metrics(before_doc_ids, q["relevant_docs"], ks=(1,))
        filtered_hits = ann.search(q["question"], k=5, where={"status": "current"})
        filtered_doc_ids = [h.doc_id for h in filtered_hits]
        after = retrieval_metrics(filtered_doc_ids, q["relevant_docs"], ks=(1,))
        print(f"{qid}: before hit_rate@1={before['hit_rate@1']:.4f}, after={after['hit_rate@1']:.4f}")

    # -----------------------------------------------------------------
    # D2 -- at ~160 chunks the exact-vs-ANN comparison above is close to
    # meaningless: HNSW's per-query graph traversal loses to one BLAS matmul
    # over a matrix this small. Scale the corpus up and find the crossover.
    #
    # This shells out to scripts/expand_corpus.py, which the lab says adds
    # filler documents with no golden answers (safe to index, excluded from
    # quality metrics here since we only ever time queries, never re-score
    # retrieval quality on the expanded corpus).
    #
    # NOTE: the exact chunk counts you get per --docs value depend on that
    # script's own logic -- treat "~4k" / "~40k" as approximate, print the
    # real chunk_count from each run, and use THAT number in your report
    # rather than the requested one.
    # -----------------------------------------------------------------
    print("\nD2 -- exact vs. ANN query latency at scale")
    scale_points = [
        (None, "~160 chunks (no expansion)"),
        (400, "~4k chunks"),
        (4000, "~40k chunks"),
    ]
    timing_rows: dict[str, dict] = {}
    sample_questions = questions[:10]  # enough to be stable, cheap enough to repeat

    for docs_arg, label in scale_points:
        if docs_arg is not None:
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/expand_corpus.py"), "--docs", str(docs_arg)],
                check=True,
            )
            filler_dir = ROOT / "data/corpus_scaled"
            filler = ({p.stem: p.read_text(encoding="utf-8")
                      for p in sorted(filler_dir.glob("*.md"))}
                     if filler_dir.exists() else {})
            print(f"  ({len(filler)} filler docs loaded from data/corpus_scaled for '{label}')")
            scaled_corpus = {**corpus, **filler}
        else:
            scaled_corpus = corpus

        scaled_chunks = build_chunks(scaled_corpus, "markdown", 800)
        exact_r = DenseRetriever(scaled_chunks, show_progress=False)
        ann_r = ChromaRetriever(
            scaled_chunks, path=".chroma",
            collection=f"lab3_scale_{docs_arg or 0}", reset=True,
        )

        exact_p50, exact_p95 = _time_queries(exact_r, sample_questions)
        ann_p50, ann_p95 = _time_queries(ann_r, sample_questions)

        timing_rows[label] = {
            "chunk_count": len(scaled_chunks),
            "exact_p50_ms": exact_p50,
            "exact_p95_ms": exact_p95,
            "ann_p50_ms": ann_p50,
            "ann_p95_ms": ann_p95,
        }

    print(table(timing_rows, cols=("chunk_count", "exact_p50_ms", "exact_p95_ms", "ann_p50_ms", "ann_p95_ms")))

    crossover = next((label for label, row in timing_rows.items() if row["ann_p95_ms"] < row["exact_p95_ms"]), None)
    if crossover:
        print(f"\nCrossover: HNSW becomes faster than exact search at {crossover}.")
    else:
        print("\nHNSW never overtakes exact search at the scales measured here.")
        print("Either widen the sweep (a larger --docs value) or report this")
        print("directly -- 'no crossover observed up to ~40k chunks' is a")
        print("legitimate, reportable finding, not a failed sweep.")


SWEEPS = {
    "chunking": sweep_chunking,
    "retrieval": sweep_retrieval,
    "rerank": sweep_rerank,
    "index": sweep_index,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--sweep", choices=list(SWEEPS))
    args = ap.parse_args()
    if args.baseline or not args.sweep:
        sweep_baseline()
    if args.sweep:
        SWEEPS[args.sweep]()


if __name__ == "__main__":
    main()