# Lab 7 Evaluation Report

## 1. What it does

Aurora answers questions using the health-policy documents, shows the passages
behind each answer, and says when it cannot find enough evidence. It records
service activity and offers a read-only tool mode; important coverage, claims,
and financial decisions still need a person to check the source.

## 2. How well it works

The offline evaluation covers 45 golden questions (40 answerable, five
unanswerable) and uses the committed 4.2 MB cache: 143 genuine model responses
and 209 embedding rows. The limits below are unchanged.

| Metric | Measured | Limit | Result |
|---|---:|---:|---|
| Correctness, answerable cases | 0.7875 | >= 0.75 | Pass |
| Faithfulness | 0.9778 | >= 0.90 | Pass |
| Citation validity | 1.0000 | >= 0.98 | Pass |
| Refusal recall | 1.0000 | >= 0.80 | Pass |
| Refusal precision | 1.0000 | >= 0.75 | Pass |
| Retrieval hit rate @ 5 | 1.0000 | >= 0.85 | Pass |
| Cost/query, offline replay | $0.0000 | <= $0.010 | Pass, cache-only |
| Gate p95, local offline replay | 7.7848 ms | <= 6,000 ms | Pass |

The [saved per-case evaluation](reports/lab7_gate.json) records 17 of 40
answerable cases with partial normalized judge scores of 0.5; none scored zero.
All five unanswerable cases were refused and no invalid citations were
recorded. These scores indicate remaining answer-quality work despite the
aggregate gate passing.

**D3 evidence.** On temporary branch `lab7-red-build-evidence-20261013`,
regression commit `fa80a44d5f529048585c05da614b8834378f277d` set `final_k=1`.
The genuine [Actions run 38444132522](https://github.com/punitgauttam/aiip-lab-submission/actions/runs/38444132522)
completed setup, cache verification, tests, and lint, then failed only at the
golden-set gate: correctness 0.6750 (< 0.75), refusal precision 0.7143
(< 0.75), and hit rate @5 0.8250 (< 0.85). The regression was reverted in
`6e786325864d3fb6b90989b832582c74c21f7177`; the subsequent
[Actions run 38446792601](https://github.com/punitgauttam/aiip-lab-submission/actions/runs/38446792601)
passed, including the gate. The genuine run-summary screenshot is
[reports/lab7_red_build.png](reports/lab7_red_build.png). GitHub's signed-out
page does not show job logs; the metric values above were verified from the
Actions job logs. `main` and `origin/main` remain at `884be825`.

## 3. Where it fails

Q23 remains a cautious, cited partial answer about physiotherapy coverage; its
normalized correctness score is 0.5 because the policy text does not establish
eligibility clearly. The refusal-recovery path fixed the earlier false
refusals on Q23 and Q44 by retrying a refusal with focused BM25 evidence and,
for a short identifier chunk, its following section. It retains the original
refusal if the retry still lacks validated support.

The semantic-cache test exposed a wrong-intent match at cosine similarity
0.9294 and a useful paraphrase at 0.9252. A threshold of 0.92 admitted the
wrong-intent match; 0.93 rejected both. The implementation keeps 0.95, which
avoided wrong hits in this sample but produced no observed semantic hits.
Exact-response caching was verified. This is a small sample, not a general
threshold guarantee.

## 4. What it costs

The isolated offline gate costs $0/query because model calls are replayed from
cache. A previously measured mixed-cache run cost $0.0002098/query. If that
same cache-hit/recovery mix persisted, the projection is $0.21 per 1,000
queries and $765.77/year at 10,000 queries/day. This is not an uncached
provider estimate or deployment budget; steady-state live cost is unknown.

## 5. How fast it is

The two-request local service sample returned validated answers for Q23 and
Q44. `/metrics` reported p50/p95/p99 of 18.97/25.42/25.42 ms; a buffered SSE
request on a model-cache hit measured 17.69 ms to first token and 18.36 ms
total. The two-request stage sample was:

| Stage | p50 | p95 |
|---|---:|---:|
| Query embedding, cache lookup | 0.01 ms | 0.01 ms |
| Dense retrieval | 2.36 ms | 6.47 ms |
| Reranking | Not used | Not used |
| Generate and validate | 9.53 ms | 10.63 ms |
| BM25 refusal recovery | 1.81 ms | 3.11 ms |
| End-to-end | 18.97 ms | 25.42 ms |

The sample size is two, so p95 is the maximum observation, not a stable SLO
estimate. Generation and validation are combined in the current trace sample;
they are not independently measured. The local offline gate recorded a p95 of 7.7848 ms; the successful remote
restoration run measured 3.0231 ms. These cache-backed results do not
establish uncached live latency. The service
buffers until citation validation, then streams answer chunks and citations;
this protects users from unvalidated text but delays TTFT until generation and
validation finish.

## 6. What it is not safe for

Do not rely on Aurora alone for coverage, claims, or financial decisions.
Retrieval can miss answerable multi-hop or paraphrased questions, and semantic
reuse can transfer an answer to a subtly different request. The policy text
may not resolve a user's circumstances; a human must inspect the cited
passages before acting.

## 7. What to do next, ranked by expected value

1. **Human-review Q23 and clarify the policy evidence if needed.** Expected
   value: high safety value; establish whether the partial answer is the right
   interpretation before presenting it as coverage guidance.
2. **Measure uncached latency and cost under representative traffic with a
   fixed spend cap.** Expected value: high decision value; determine whether
   the live service meets the 6-second p95 and $0.01/query limits, which
   cache-backed measurements cannot establish.
3. **Add independently labeled paraphrase and multi-hop cases to the golden
   set.** Expected value: medium quality value; make retrieval blind spots
   visible in the gate rather than relying on the current 45-question sample.

Local verification after restoration: 11 Lab 7 tests passed, Ruff passed, and
the offline gate passed all eight thresholds. CI independently passed the
cache check, prior-lab interface check, Lab 7 tests, Ruff, and the offline
gate on the restored temporary branch.
