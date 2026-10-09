# Lab 7 Evaluation Report

## 1. What the system does

The Aurora assistant searches the policy corpus, answers through the Lab 4 RAG
pipeline (including the active Lab 5 answer-format and citation checks), and
returns expandable source excerpts. A FastAPI service exposes answers, health,
metrics, and buffered server-sent events. Tool mode reuses the Lab 6 agent with
all five defenses, read-only tools, and refund confirmation denied.

The current index builder is the existing Lab 4 `build_retriever()` (dense
Markdown chunks of size 800). No saved Lab 3 sweep-winner artifact was present,
so this report does not claim that this configuration was the Lab 3 winner.

## 2. Golden-set results

Offline gate run: 45 golden cases, using the isolated 4.2 MB Lab 7 cache
(143 genuine chat responses and 209 genuine embedding rows). Values below are
measured, not manually entered. The unchanged starter limits are shown.

| Metric | Measured | Limit | Result |
|---|---:|---:|---|
| Correctness (answerable cases) | 0.7875 | >= 0.75 | Pass |
| Faithfulness | 0.9778 | >= 0.90 | Pass |
| Citation validity | 1.0000 | >= 0.98 | Pass |
| Refusal recall | 1.0000 | >= 0.80 | Pass |
| Refusal precision | 1.0000 | >= 0.75 | Pass |
| Retrieval hit rate @ 5 | 1.0000 | >= 0.85 | Pass |
| Cost/query, offline replay | $0.0000 | <= $0.010 | Pass, cache-only |
| Gate p95 latency | 8.72 ms | <= 6,000 ms | Pass, offline replay |

The earlier gate returned false refusals for Q23 and Q44 because its dense
top-five chunks did not present the relevant evidence together: Q23 included
teleconsultation and unrelated Bronze sections, while Q44 retrieved the Silver
identifier chunk but not its adjacent sum-insured section. Lab 4 correctly
fell back to refusal when those contexts could not support an answer. That
run had two false refusals; the current run has zero, with refusal precision
improving from 0.7143 to 1.0000 under the same threshold.

Lab 7 now retries only a validated refusal using the top BM25 evidence. If that
hit is a short document-identifier chunk, it includes the next section from
that same document. The retry still uses Lab 4 citation validation, and an
unvalidated or refused retry preserves the original refusal. No case IDs or
thresholds are special-cased. Q23 now returns a cautious, cited partial answer
that says the source does not specifically establish physiotherapy coverage;
its normalized judge score is 0.5, so the uncertainty remains visible. Q44
answers the sum-insured options with two citations. All five unanswerable cases
remain refused, with zero false refusals and no invalid citations.

## 3. Cache and service measurements

Four different-answer question pairs were embedded. Their cosine similarities
were 0.8782 (Silver/Gold room limits), 0.8479 (Bronze/Gold maternity), 0.9294
(claim deadline after discharge/after admission), and 0.7194 (cataract/no-claim
bonus). With the actual cache lookup, the 0.9294 pair hit at threshold 0.92 and
did not hit at 0.93 or 0.95. Four answer-equivalent paraphrases scored 0.7681,
0.8849, 0.9252, and 0.8456. Thus, this sample has no threshold that accepts the
tested paraphrase at 0.9252 without also admitting the 0.9294 wrong-intent
match. The implementation conservatively retains 0.95; semantic hits were not
observed for this sample, while exact response caching was verified.

The service starts with an index of 164 chunks. In a two-request isolated
offline sample, `/ask` returned 200 with validated citations for Q23 and Q44;
the individual reported latencies were 23.19 and 18.60 ms. `/metrics`
reported p50/p95/p99 of 18.97/25.42/25.42 ms, zero cost, zero errors, and
zero response-cache hits. A prior single exact-cache repeat took 0.07 ms;
this is not a cached p95. A buffered SSE request on a model-cache hit
measured 17.69 ms to first token and 18.36 ms total. These are cache-backed
local timings, not live-provider SLO evidence.

The traced Q23/Q44 requests in a separate two-request sample yielded:

| Stage | p50 | p95 |
|---|---:|---:|
| Query embedding (cache lookup) | 0.01 ms | 0.01 ms |
| Dense retrieval | 2.36 ms | 6.47 ms |
| Reranking | Not used | Not used |
| Generate + validate | 9.53 ms | 10.63 ms |
| BM25 refusal recovery | 1.81 ms | 3.11 ms |
| End-to-end | 18.97 ms | 25.42 ms |

Sample sizes are two; p95 here is the maximum observation, not a stable SLO
estimate. Profile live generation under representative traffic before choosing
production latency optimizations.

Streaming buffers until citation validation, then emits answer chunks followed
by citations. This prevents clients displaying unvalidated text, but sacrifices
live-generation TTFT. Stage and TTFT values above are cache-backed only.

## 4. Cost arithmetic and limitations

The isolated offline run measured $0/query because every model operation was
replayed from cache. The first full gate after recovery, run against a mixed
development cache, measured $0.0002098/query and 3,419.31 ms p95. If that
same cache-hit/recovery mix persisted, the projection is $0.21 per 1,000
queries and $765.77/year at 10,000 queries/day (3.65 million/year). This is a
cache-dependent projection, not an uncached provider estimate or deployment
budget; live uncached p95 and steady-state cost are not established.

Do not rely on this assistant alone for coverage, claims, or financial
decisions. Retrieval can miss answerable multi-hop or paraphrased questions,
and semantic reuse can transfer an answer to a subtly different request.
Require human review for consequential decisions.

## 5. Ranked next steps

1. Evaluate the cautious Q23 partial answer with a human label; decide whether
   the corpus should state physiotherapy eligibility more explicitly.
2. Measure uncached latency and per-query cost under representative live
   traffic; current headline timings are cache-backed.
3. Make the existing Lab 3, Lab 4, and Lab 6 source modules available in a
   clean checkout, then rerun CI. The first remote run failed during test
   collection because `labs.lab3` and `labs.lab4` are absent from GitHub's
   checkout. The local red-gate text output is not the required screenshot.

Final local verification: the full offline test suite passed (44 tests), the
Lab 7 tests passed (11 tests), Ruff passed, and the isolated-cache golden gate
passed all eight unchanged thresholds across 45 cases. The genuine cache
bundle is in `labs/lab7/offline_cache/calls.sqlite3`; it contains actual
responses and embeddings selected from the development cache, not generated
placeholder rows. A separate controlled red-gate probe also exited non-zero;
its output is in [the red-gate evidence](reports/lab7_gate_red_probe.txt).
GitHub Actions runs
[37962116366](https://github.com/punitgauttam/aiip-lab-submission/actions/runs/37962116366)
and
[37964718528](https://github.com/punitgauttam/aiip-lab-submission/actions/runs/37964718528)
passed dependency installation and cache validation, then failed collecting
the Lab 7 tests with `ModuleNotFoundError` for `labs.lab3` and `labs.lab4`;
lint and gate steps were skipped. Those source directories exist only in the
local untracked checkout and were not added or pushed. There is no remote
green run or red-build screenshot.
