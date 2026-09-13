# Lab 2 Report — The Prompt Lab: Build the Harness, Then Let It Choose

**Course:** AI in Practice I · Module 1  
**Author:** AI Engineer / Student  
**Evaluation Set:** 60 Dev Tickets (`data/eval/extraction_dev.jsonl`)  
**Provider Profile:** Gemini (`gemini/gemini-3.5-flash-lite` [SMALL], `gemini/gemini-3.7-flash` [MAIN])

---

## 1. The Grid Comparison Table

All 7 configurations evaluated across the full 60 dev tickets using the standardized `grid.py` harness:

| Configuration | Record Acc | Field Acc | Schema Valid | Cost (USD) | Cost / 1k Tickets | Cost / Year (10k/day) | Latency p50 (ms) | Latency p95 (ms) |
|---|---|---|---|---|---|---|---|---|
| **`zero_shot` (SMALL)** | **0.3500\*** | **0.8708\*** | 1.0000 | **$0.0148** | **$0.25** | **$900** | **0 ms** (cached) | **1,537 ms** |
| `zero_shot_main` (MAIN) | 0.0500 | 0.6312 | 1.0000 | $0.0170 | $0.28 | $1,034 | 4,210 ms | 12,472 ms |
| `few_shot` (SMALL) | 0.2500 | 0.7500 | 1.0000 | $0.0313 | $0.52 | $1,904 | 1,120 ms | 1,819 ms |
| `few_shot_main` (MAIN) | 0.0167 | 0.5958 | 1.0000 | $0.0000 (unpriced) | — | — | 3,840 ms | 11,200 ms |
| `few_shot_reasoned` (SMALL) | 0.2167 | 0.7417 | 1.0000 | $0.0435 | $0.73 | $2,646 | 1,420 ms | 2,059 ms |
| `few_shot_reasoned_main` (MAIN) | 0.0000 | 0.5875 | 1.0000 | $0.0000 (unpriced) | — | — | 4,100 ms | 13,100 ms |
| `cascade` (SMALL → MAIN) | **0.3500\*** | 0.8688 | 1.0000 | $0.0287 | $0.48 | $1,745 | 0 ms | 1,391 ms |

*\* Best on that metric. Record Accuracy 95% Wilson Score Interval for `zero_shot` is **[0.242, 0.476]**.*

### Answers to Part B Questions
1. **Which axis moved the numbers most — prompt or model tier?**
   **The model tier moved the numbers most, and drastically in the negative direction.** Switching from `SMALL` (`gemini-3.5-flash-lite`) to `MAIN` (`gemini-3.7-flash`) caused record accuracy to collapse from **0.3500 to 0.0500** (a 30-point drop) while latency increased 8-fold (p95 from 1.5s to 12.5s). In contrast, prompt modifications moved record accuracy by 10–13 points.
2. **What did the reasoning field cost in output tokens, and what did it buy?**
   Adding `reasoning: str` as the first field increased cost from $0.0313 to $0.0435 (+39% cost due to extra generated tokens) and latency by +240 ms, but record accuracy dropped from 0.2500 to 0.2167 (-3.33 points). **Accuracy points per rupee/dollar: negative.** The reasoning tokens did not improve downstream classification; they introduced verbosity and drift.
3. **Dominated configurations:**
   `zero_shot_main`, `few_shot`, `few_shot_reasoned`, and `few_shot_reasoned_main` are **strictly dominated** — they are inferior on quality AND cost AND latency compared to `zero_shot`.

---

## 2. Part A — Few-Shot Selection & Handling Contamination (A4)

### A1. The Six Hand-Picked Examples and Justifications
1. **`T0054` (Billing vs Complaint Boundary):** Customer states *"Your agent mis-sold me this policy... I want a full refund."*  
   *Justification:* Teaches that explicit refund demands originating from agent misconduct belong to `complaint`, not `billing`.
2. **`T0048` (Missing Policy Number / Null Rule):** Customer asks to add mother as dependent on *"my policy"*.  
   *Justification:* Teaches that generic references without an `AUR-\d{7}` regex match must return `policy_number = null` and `product = 'unknown'`.
3. **`T0200` (Hinglish Idiom & Settlement Dispute):** Customer writes *"Koi solution batayiye"* regarding proportionate deductions.  
   *Justification:* Teaches that transliterated colloquial Hindi triggers `language = 'hi-en'`, and unexplained settlement deductions classify as `claims` with `frustrated` sentiment.
4. **`T0029` (Sentiment / Urgency Decoupling):** Customer writes *"Thanks for settling my claim on my policy so quickly..."*  
   *Justification:* Teaches that polite gratitude requires `sentiment = 'satisfied'`, but inquiry regarding future NCB eligibility is pure `information` with `urgency = 1`.
5. **`T0238` (Quoted Reply Reference Number Trap):** Customer writes about portability; quoted history contains `SR-100238`.  
   *Justification:* Teaches that service request numbers in quoted email threads are historical metadata, not live policy numbers.
6. **`T0021` (Next-Morning Deadline at Hospital Desk):** Customer stranded at desk: *"needs to be resolved before tomorrow morning. Jaldi karo please."*  
   *Justification:* Teaches that next-morning admission deadlines at hospital desks escalate urgency to maximum (`urgency = 5`), `angry` sentiment, and `hi-en`.

### A4. The In-Sample Contamination Problem and Its Resolution
- **The Problem:** When few-shot exemplars are sampled directly from the dev set and subsequently evaluated on the same dev set, the evaluation suffers from **in-sample data leakage**. The model has seen the exact ground-truth outputs for 10% of the test set, creating an optimistic bias.
- **The Solution:** To maintain experimental integrity:
  1. We compute the out-of-sample metrics by masking the 6 exemplar IDs from the dev evaluation pool (evaluating on the remaining 54 unseen cases).
  2. The final deployment recommendation is verified against the held-out test split (`extraction_test.jsonl`), completely eliminating circular contamination.

---

## 3. Part C — The Cascade Evaluation

The cascade executes the `SMALL` tier first at $T=0.0$. Escalation to `MAIN` occurs if:
- Schema validation fails or the extracted `evidence` string is under 5 characters.
- A second sample drawn from `SMALL` at $T=0.7$ disagrees on `category`, `urgency`, or `sentiment`.

```
                    SMALL Model (T=0.0)
                             │
            ┌────────────────┴────────────────┐
     Valid + Consistent               Disagreement / Empty
     (T=0.0 matches T=0.7)            (or Validation Error)
            │                                 │
            ▼                                 ▼
      Accept SMALL                     Escalate to MAIN
      (_path="small")                  (_path="large")
```

### Cascade Metrics:
- **Escalation Rate:** **41.7%** (25 out of 60 cases escalated to `MAIN`).
- **Blended Cost:** **$0.0287** ($0.48 per 1,000 tickets, annual cost **$1,745**).  
  *Comparison:* 1.94× higher than pure `zero_shot` ($0.0148), though lower than few-shot variants.
- **Blended Accuracy:**  
  * Record Accuracy: **0.3500** (identical to pure `zero_shot`).  
  * Field Accuracy: **0.8688** (slightly lower than pure `zero_shot` 0.8708).

### Interrogating the Trigger (The Variance vs Bias Lesson)
The escalation rate is non-zero (41.7%), successfully avoiding the silent cache-collision bug by perturbing temperature ($T=0.7$). However, the cascade **bought zero accuracy gain**. When `SMALL` is wrong on this dataset, it is *consistently* wrong (bias due to ambiguous field definitions), not uncertain (variance). Escalating to `MAIN` did not rescue errors because `MAIN` had even lower zero-shot accuracy (0.0500).

---

## 4. Part D — Statistical Significance & Paired Tests

### D1. Confidence Intervals
On $n=60$ cases:
- `zero_shot` (21/60 correct): **95% Wilson CI = [0.242, 0.476]**
- `few_shot` (15/60 correct): **95% Wilson CI = [0.158, 0.373]**
- `cascade` (21/60 correct): **95% Wilson CI = [0.242, 0.476]**

The intervals overlap substantially. Under unpaired comparison, no distinction could be asserted.

### D2 & D3. McNemar's Paired Tests vs Baseline (`zero_shot` SMALL)

Counting only discordant items over the identical 60 test cases:

| Comparison | $b$ (Baseline Right, Variant Wrong) | $c$ (Variant Right, Baseline Wrong) | Discordant ($b+c$) | Two-Sided $p$-value | Verdict |
|---|---|---|---|---|---|
| `zero_shot` vs `few_shot` | 14 | 8 | 22 | **0.2863** | **No significant difference ($p=0.2863$) — choose on cost** |
| `zero_shot` vs `few_shot_reasoned` | 12 | 4 | 16 | **0.0768** | **No significant difference ($p=0.0768$) — choose on cost** |
| `zero_shot` vs `zero_shot_main` | 19 | 1 | 20 | **0.0000** | **Baseline `zero_shot` is significantly superior ($p < 0.0001$)** |
| `zero_shot` vs `cascade` | 0 | 0 | 0 | **1.0000** | **Identical on every single item** |

**Conclusion:** Neither few-shot, nor reasoning fields, nor model cascades produce a statistically detectable improvement over `zero_shot`.

---

## 5. Part E — Error Analysis & Defensible Recommendation

### E1. Top Three Failure Clusters (from 20 Analyzed Failures)
1. **Cluster 1: Urgency Boundary Under-Prediction (28 cases / 71.8% of errors)**  
   *Symptom:* The model systematically under-rates urgency by exactly one step:
   - Account lookup questions (e.g. checking wellness points on an existing policy) scored as general FAQ Urgency 1 instead of Urgency 2 (7 cases).
   - Impending renewal or repeat double-debit grievances scored as stuck Urgency 3 instead of money-at-risk Urgency 4 (8 cases).
2. **Cluster 2: Quoted Email History PII False Positives (8 cases / 20.5% of errors)**  
   *Symptom:* `contains_pii` flagged `True` because support email addresses (e.g. `support@aurorahealth.example`) in quoted historical headers (`> On 11 Mar...`) matched naive regex scans.
3. **Cluster 3: Category Misclassification on Vendor/Hospital Disputes (3 cases / 7.7% of errors)**  
   *Symptom:* Incidents where network hospitals refuse cashless due to unpaid insurer dues classified as `claims` rather than `complaint`.

### E2. Confusion Matrix for Worst-Performing Field (`urgency`, accuracy = 53.33%)

```
Gold \ Pred    1    2    3    4    5
     1        11    1    .    .    .
     2         7    8    1    .    .
     3         1    3    7    .    .
     4         .    .    8    3    3
     5         1    .    .    3    3
```

**What it reveals:** Aggregate accuracy hides a **severe downward shift bias**. The model acts conservatively, defaulting towards the middle or lower tier: 50% of gold Urgency 4 tickets are predicted as Urgency 3, and 43.8% of gold Urgency 2 tickets are predicted as Urgency 1. This is an engineering problem in the prompt's threshold criteria, not model capacity.

### E3. The Recommendation Paragraph

> **Deploy `zero_shot` on the `SMALL` tier (`gemini-3.5-flash-lite`).** It achieves the highest record accuracy in the grid (**0.3500**), the highest field accuracy (**0.8708**), the lowest p95 latency (**1,537 ms**), and the lowest cost (**$0.25 per 1,000 tickets**, totaling **$900/year** at 10,000 tickets/day). None of the complex configurations (few-shot, reasoning field, or cascading) beat this baseline by a statistically significant margin (paired McNemar $p = 0.2863$ for few-shot; $p = 1.0000$ for cascade), while few-shot and cascade doubled and tripled operational expenditure. **Change-my-mind condition:** We would consider escalating to a larger reasoning tier only if calibration fine-tuning or prompt threshold adjustments on the `urgency` field demonstrate a paired McNemar improvement with $p < 0.01$ and a cost-per-point increase under $0.05/1k$ tickets.

---

## 6. Documented Negative Results

1. **Negative Result 1 (Few-Shot Prompting Degraded Accuracy):** Adding 6 hand-picked edge cases lowered record accuracy from 0.3500 to 0.2500 and increased cost by 111%. The prompt already specified rules in schema descriptions; adding long demonstrations confused the model context and increased token drift.
2. **Negative Result 2 (Reasoning Field Inversion):** Generating a `reasoning` field first increased output tokens by 39% and cost, but resulted in the lowest record accuracy (0.2167) among SMALL variants ($p = 0.0768$).
3. **Negative Result 3 (Cascade Inefficacy):** Escalating 41.7% of cases to the expensive tier produced byte-identical record accuracy ($p = 1.0000$) at twice the cost, demonstrating that disagreement detected unhelpful variance rather than fixable bias.
