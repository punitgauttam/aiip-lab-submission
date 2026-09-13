# Comprehensive Problem-Solving Guide & Summary: Lab 2 (The Prompt Lab)

**Document Purpose:** This document provides an exhaustive, plain-language walkthrough of how Lab 2 was approached, executed, analyzed, and synthesized. It explains the architectural choices, statistical methods, empirical findings, and connections to **Theory 1 through Theory 4** (especially **T3: Evaluation-Driven Development** and **T4: Retrieval Engineering**).

---

## 1. Executive Overview & The Central Thesis

In Lab 1, an extraction pipeline was built and verified. However, in real-world AI engineering, having code that works is only the starting point. When an executive or engineering manager asks:
> *"Which model and prompting technique should we deploy to production? What will it cost us per year at 10,000 tickets per day?"*

An answer like *"this prompt felt better"* or *"obviously the bigger, expensive model is superior"* is unacceptable and dangerous. 

### The Core Result of Lab 2
We benchmarked **7 distinct configurations** across prompt strategies (zero-shot, few-shot, few-shot + reasoning) and model tiers (`SMALL` vs `MAIN`) in an automated evaluation harness on 60 customer support tickets. 

**The empirical finding:**
> **The simplest, cheapest baseline (`zero_shot` on `gemini-3.5-flash-lite`) outperformed every complex configuration.** It scored the highest record accuracy (**0.3500**), the highest field accuracy (**0.8708**), the lowest p95 latency (**1,537 ms**), and the lowest cost (**$0.25 per 1,000 tickets**, or **$900/year** at 10,000 tickets/day).
>
> Adding few-shot examples lowered accuracy (0.2500) and doubled cost ($1,904/year). Adding a reasoning field lowered accuracy further (0.2167) and tripled cost ($2,646/year). Switching to the larger `MAIN` model (`gemini-3.7-flash`) caused record accuracy to collapse to **0.0500** while increasing latency by **800%** (12.5 seconds per call).
>
> In accordance with statistical testing (McNemar's paired test, $p = 0.2863$ for few-shot, $p = 1.0000$ for cascade), **none of the complex configurations beat the baseline by a detectable margin. Therefore, the defensible engineering decision is to ship the cheap baseline.**

In AI engineering, **negative results are not failures; they are the most valuable business findings.** They prevent companies from wasting hundreds of thousands of dollars on complex architectures that add zero value.

---

## 2. Step-by-Step Approach & Methodology

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          THE EXPERIMENTAL PIPELINE                      │
└─────────────────────────────────────────────────────────────────────────┘
  1. HARNESS PREPARATION      2. FEW-SHOT SELECTION       3. GRID EXECUTION
  • Clean env validation      • Pick 6 distinct edges     • Run 7 configurations
  • Cache SQLite integration  • Byte-identical JSON       • Record acc, field acc,
  • Zero Lab 1 modifications  • Solve A4 data leakage       cost, and latency
              │                           │                        │
              └───────────────────────────┼────────────────────────┘
                                          ▼
  4. THE CASCADE              5. STATISTICAL TESTS        6. ERROR TRIAGE
  • SMALL (T=0.0)             • Wilson Score 95% CIs      • Read 20 failures
  • Compare SMALL (T=0.7)     • McNemar's Paired Test     • Confusion matrix
  • Non-zero escalation rate  • Disprove noise claims     • Deployment decision
```

### Step 1: Environment & Tooling Integrity
Before running evaluations, we ensured the experimental harness was reproducible and isolated:
1. **Lab 1 File Protection:** The instructions strictly forbid modifying any files in `labs/lab1/`. All implementations were strictly encapsulated in `labs/lab2/variants.py` and evaluation scripts.
2. **Content-Addressed SQLite Cache:** Model calls are cached in `.aip_cache/calls.sqlite3` keyed by SHA-256 hashes of `(model, messages, temperature, schema)`. This ensures fast, reproducible runs without incurring unnecessary API costs on repeat calls.
3. **Encoding & System Stability:** Windows console encoding defaults to `cp1252`, which crashes on Unicode box-drawing characters (`──`) and emojis in support tickets. We ensured UTF-8 execution throughout.

---

### Step 2: Part A — Few-Shot Example Selection (A1 & A2) & Contamination (A4)

Few-shot prompting ("show, don't tell") is often treated as a magic wand. But few-shot examples cost input tokens on **every single API call**, forever. They must earn their keep.

#### A1. Why We Picked These Six Examples
As taught in **T2 §2.2**, picking average or typical tickets is useless; the model already knows how to handle simple cases. Exemplars must be **edge cases** where prose instructions are ambiguous:
1. **`T0054` (Billing vs. Complaint Boundary):**
   - *Text:* *"Your agent mis-sold me this policy. Nobody told me maternity has a 5-month waiting period... I want a full refund."*
   - *Why:* A naive model sees *"refund"* and classifies it as `billing`. This example teaches that when the root cause is agent misconduct or mis-selling, `complaint` overrides `billing`.
2. **`T0048` (The Null Policy Number Rule):**
   - *Text:* *"Please add my mother as a dependent on my policy."*
   - *Why:* The user writes *"my policy"* but provides no alphanumeric identifier. Teaches the model not to hallucinate or extract `"my policy"`, but strictly output `null` and `product = 'unknown'`.
3. **`T0200` (Hinglish & Prior Settlement Dissatisfaction):**
   - *Text:* *"My claim on AUR-9674338 was settled at Rs 41800 but the hospital bill was much higher... Koi solution batayiye."*
   - *Why:* Teaches that colloquial transliterated Hindi phrases (`"Koi solution batayiye"`) require `language = 'hi-en'`, and unexplained deductions represent a prior settlement failure (`sentiment = 'frustrated'`, `urgency = 3`).
4. **`T0029` (The Sentiment vs. Urgency Trap):**
   - *Text:* *"Thanks for settling my claim on my policy so quickly. Just wanted to confirm whether my no-claim bonus is affected by this claim."*
   - *Why:* Teaches that polite praise requires `sentiment = 'satisfied'`, but because it is a routine inquiry with no impending crisis, it is `category = 'information'` and `urgency = 1`.
5. **`T0238` (Quoted Email History Trap):**
   - *Text:* User asks about portability; quoted support reply contains `SR-100238`.
   - *Why:* Teaches that service ticket IDs in quoted text (`> ...`) are not policy numbers and must be ignored.
6. **`T0021` (Next-Morning Deadline at Hospital Desk):**
   - *Text:* *"Your portal has been down all morning and I am standing at the hospital insurance desk... This needs to be resolved before tomorrow morning. Jaldi karo please."*
   - *Why:* Teaches that an immediate next-morning hospital deadline pushes `urgency = 5`, `sentiment = 'angry'`, and `language = 'hi-en'`.

#### A2. Byte-Identical Formatting
In `variants.py::few_shot_block()`, we ensured that the rendered examples output JSON matching the exact Pydantic schema (`TicketRecord`) byte-for-byte. A mismatch between few-shot example structure and target output structure is one of the most common causes of schema parse failures in LLMs.

#### A4. The In-Sample Contamination Trap & The Fix
- **The Problem:** The 6 examples were drawn from the 60 dev tickets. Evaluating few-shot performance on the same 60 dev tickets is **in-sample data leakage** (test-train contamination). The model has seen the ground-truth answers for 10% of the evaluation set, artificially inflating the score.
- **The Fix:** We address this in two ways:
  1. Compute the out-of-sample metrics by masking the 6 exemplar tickets from dev evaluation (scoring only the remaining 54 unseen cases).
  2. Confirm our final recommendation against the untouched test split (`extraction_test.jsonl`), where none of the dev exemplars exist.

---

### Step 3: Part B — Running the 7-Configuration Grid

The 7 configurations in the experiment grid represent three key architectural axes:
1. **Prompt Strategy:** Zero-shot vs. Few-shot (6 examples) vs. Few-shot + Reasoning Field.
2. **Model Tier:** `SMALL` (`gemini-3.5-flash-lite`, fast/cheap) vs. `MAIN` (`gemini-3.7-flash`, reasoning model).
3. **Routing:** Single-call vs. Small→Large Cascade.

#### Field Ordering in Reasoning Models (T2 §3.3)
In `TicketRecordReasoned`, we placed `reasoning: str` as the **first** field:
```python
class TicketRecordReasoned(TicketRecord):
    reasoning: str = Field(default="", description="Step-by-step reasoning...")
```
*Why this matters:* Language models generate tokens sequentially from left to right.
- If `reasoning` is generated **first**, the model's intermediate thoughts condition its subsequent classification tokens (Chain-of-Thought prompting).
- If `reasoning` is generated **last**, the model commits to a classification first and uses the reasoning field merely to post-hoc rationalize its prior decision.

#### The Triple Metric: Quality × Cost × Latency (T1 §2, T3 §3.2)
Never measure quality alone. Our grid recorded:
- **`record_accuracy`**: Fraction of tickets where *every single field* is 100% correct (the business metric).
- **`field_accuracy`**: Fraction of individual fields correct across all tickets (the engineering metric).
- **`cost_usd`**: Total API spend across the run, scaled to **cost per 1,000 tickets** and **annual cost at 10,000 tickets/day**.
- **`p50` & `p95` Latency**: Median and 95th-percentile response times in milliseconds.

#### Key Takeaway from the Grid: Dominated Configurations
A configuration is **dominated** if another configuration exists that is better (or equal) on **Quality AND Cost AND Latency**.
- `zero_shot_main`, `few_shot`, `few_shot_reasoned`, and `few_shot_reasoned_main` were **strictly dominated** by `zero_shot` (SMALL).
- In a business setting, dominated configurations should be immediately discarded.

---

### Step 4: Part C — The Cascade Architecture

A cascade attempts to get the speed and price of a small model with the accuracy of a large model:

```
                            Customer Ticket
                                  │
                                  ▼
                      SMALL Model Call (T=0.0)
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
       Valid & Non-Empty Evidence?         Validation Failed /
                  │                        Evidence Empty (<5 chars)
                  ▼                               │
       Sample 2 at T=0.7 from SMALL               │
                  │                               │
         ┌────────┴────────┐                      │
         ▼                 ▼                      │
     Do Sample 1       Do Sample 1                │
    and 2 Agree?      and 2 Disagree?             │
         │                 │                      │
       (Yes)              (No)                    │
         │                 │                      │
         ▼                 └──────────────┬───────┘
    Accept SMALL                          ▼
   (_path='small')                Escalate to MAIN Model
   Cost: $0.25/1k                 (_path='large')
                                  Cost: $0.48/1k (Blended)
```

#### The Silent Caching Bug
If you attempt to check agreement by calling the SMALL model twice with the exact same prompt at `temperature=0.0`, **LiteLLM's cache intercepts the second request and returns the exact cached result of the first request.** The two outputs are byte-identical, disagreement is never detected, escalation rate is 0.0%, and nothing errors.
*The Fix:* Draw the second sample with `temperature=0.7`. This changes the sampling distribution and generates a different cache key.

#### The Critical Insight: Variance vs. Bias
Our cascade achieved a healthy **41.7% escalation rate** (25 of 60 tickets escalated).
However, **the cascade did not improve accuracy at all (record accuracy stayed 0.3500)** while increasing blended cost from $0.25 to $0.48 per 1,000 tickets (+92% cost).
*Why?*
Self-consistency (sampling twice to check agreement) detects **variance** (model uncertainty). But the errors in our extractor were caused by **bias** (the model consistently misunderstanding the subtle criteria of `urgency`). The model was not uncertain; it was *confidently wrong*. When an error is due to bias, asking the model twice or escalating to a model with the same prompt biases changes nothing.

---

### Step 5: Part D — Statistical Significance & Paired Testing (T3 §4)

Suppose Configuration A gets 35% and Configuration B gets 38% on 60 tickets. Is B better? **No.**

#### D1. Wilson Score Confidence Intervals
Accuracy measured on a sample is merely a point estimate. Using the Wilson score interval:
- For `zero_shot` (21/60 correct): **95% CI = [0.242, 0.476]**
- For `few_shot` (15/60 correct): **95% CI = [0.158, 0.373]**

The two intervals overlap heavily. From unpaired percentages alone, it is impossible to assert any meaningful difference.

#### D2. McNemar's Paired Test
Both configurations were tested on the **exact same 60 tickets**.
In unpaired tests, the dominant source of variance is **ticket difficulty** (some tickets are easy, some are inherently ambiguous). Comparing overall percentages forces you to fight this noise.
By pairing tickets one-to-one, between-item variance cancels out completely! We count only **discordant pairs**:
- $b$: Number of tickets where Baseline is **Right** and Variant is **Wrong**.
- $c$: Number of tickets where Variant is **Right** and Baseline is **Wrong**.

Under the null hypothesis $H_0$ that both systems are equally good, discordant items are a fair 50/50 coin flip. We compute the exact two-sided binomial probability:
$$\text{p-value} = 2 \times \sum_{k=0}^{\min(b,c)} \binom{b+c}{k} 0.5^{b+c}$$

#### The Paired Results:
- `zero_shot` vs `few_shot`: $b=14, c=8 \implies p = 0.2863$.
- `zero_shot` vs `few_shot_reasoned`: $b=12, c=4 \implies p = 0.0768$.
- `zero_shot` vs `cascade`: $b=0, c=0 \implies p = 1.0000$.
- `zero_shot` vs `zero_shot_main`: $b=19, c=1 \implies p = 0.0000$.

**Conclusion:** At the standard significance threshold ($\alpha = 0.05$), neither few-shot nor reasoning nor cascading provides any statistically significant improvement over `zero_shot`. The rule of T3 holds: **"No number, no claim. When differences are not statistically significant, choose on cost."**

---

### Step 6: Part E — Error Analysis & The Hidden Systematic Confusion

An aggregate accuracy percentage tells you *how much* is wrong, never *what* is wrong. To improve a system, you must read actual failures.

#### E1. Analysis of 20 Failing Records: The Three Error Clusters
We analyzed the 39 failing tickets in `zero_shot`:
1. **Cluster 1: Urgency Boundary Under-Prediction (28 tickets / 71.8% of errors)**
   - The model systematically under-predicts urgency by 1 level.
   - *Example:* Ticket `T0095` asks for wellness points on an existing policy. To answer, Aurora must look up the customer's account (Gold rule: Urgency = 2). The model saw a general question and classified it as Urgency = 1.
   - *Example:* Ticket `T0128` involves a double debit with impending deadline (Gold rule: Urgency = 4). The model classified it as Urgency = 3.
2. **Cluster 2: Quoted Email History False-Positive PII (8 tickets / 20.5% of errors)**
   - In `extract_deterministic`, regex searched the entire ticket body without stripping quoted historical text. When support staff replied (`> On 11 Mar 2026, Aurora Support <support@aurorahealth.example> wrote:`), the email regex flagged `contains_pii = True`.
3. **Cluster 3: Vendor / Hospital Refusal Dispute (3 tickets / 7.7% of errors)**
   - In `T0025`, a network hospital refused cashless service because Aurora had not cleared its dues to the hospital. The model classified this as `claims` because "cashless" was mentioned, whereas the true label is `complaint` (the grievance is against Aurora's commercial conduct).

#### E2. The Confusion Matrix for Urgency (Accuracy = 53.33%)

```
                  Predicted Urgency
                 1     2     3     4     5
Gold Urgency  ───────────────────────────────
     1       │  11     1     .     .     .
     2       │   7     8     1     .     .   <-- 43.8% under-predicted as 1
     3       │   1     3     7     .     .   <-- 25.0% under-predicted as 2
     4       │   .     .     8     3     3   <-- 57.1% under-predicted as 3
     5       │   1     .     .     3     3   <-- 50.0% under-predicted as 4 or 1
```

**What the confusion matrix reveals:**
The aggregate field accuracy of 87% completely masks a **severe downward shift bias in urgency**. The model consistently hesitates to assign higher urgency levels. It treats account-lookup requests as general FAQs (2 $\to$ 1) and treats urgent financial grievances as routine stuck tickets (4 $\to$ 3). 
This is a high-leverage finding: fixing the prompt's threshold criteria for urgency alone would increase record accuracy from **35% to over 70%** without changing models or paying a cent more!

---

## 3. Deep Connections to Module 1 Theory (T1 – T4)

| Theory Concept | Location in Lectures | How It Directly Appeared in Lab 2 |
|---|---|---|
| **The Four Resources** | **T1 §2** | Tokens, money, latency, and attention. The grid directly traded tokens and money against accuracy. |
| **Model Economics & Thinking Tokens** | **T1 §5** | `MAIN` (`gemini-3.7-flash`) spends un-disableable thinking tokens that blow up latency to 12.5 seconds, while `SMALL` (`gemini-3.5-flash-lite`) spent 0 thinking tokens and cost 10x less. |
| **Field Ordering Matters** | **T2 §3.3** | Declaring `reasoning: str` first in `TicketRecordReasoned` ensures that generation conditions on thoughts rather than post-hoc rationalizing. |
| **Few-Shot: Pick Edges, Not Averages** | **T2 §2.2** | Our 6 exemplars focused on boundary conditions (billing vs complaint, quoted replies, Hinglish, null policy). |
| **Prompt Descriptions Do the Heavy Lifting** | **T2 §3.2** | Why did few-shot buy nothing? Because our Pydantic field `description`s already contained the exact classification rules. The few-shot examples had nothing new to teach. |
| **Golden Set Discipline & Contamination** | **T3 §2.2, §2.4** | Addressed in Part A4: evaluating few-shot exemplars on the dev set causes in-sample contamination that must be controlled. |
| **Field Accuracy vs. Record Accuracy** | **T3 §3.1** | Field accuracy was 87.1%, but record accuracy was 35.0%. Six independent 95% fields yield $0.95^6 \approx 73\%$. The business cares about record accuracy because one wrong field requires a human in the loop. |
| **Statistical Honesty: No Number, No Claim** | **T3 §4** | Computing Wilson score intervals and McNemar's exact binomial test to prove that differences between variants were statistical noise. |
| **Cascades & Disagreement Signal** | **T2 §4, T3 §5** | Proving that disagreement detects *variance*, but real errors stem from *bias*. Cascading cannot fix bias. |
| **Retrieval Engineering Preview** | **T4 §1, §4** | Selecting few-shot examples dynamically is fundamentally a retrieval problem (previewing Lab 3). |

---

## 4. Final Recommendation & Business Impact

### The Recommendation
> **Deploy `zero_shot` on `gemini-3.5-flash-lite` (`SMALL`).**
> - **Record Accuracy:** 0.3500 (95% CI: [0.242, 0.476])
> - **Field Accuracy:** 0.8708
> - **p95 Latency:** 1,537 ms
> - **Cost per 1,000 Tickets:** $0.25
> - **Annual Operating Cost at 10,000 tickets/day:** **$900 USD/year**

### Comparison with the Intuitive "Bigger is Better" Alternative (`zero_shot_main`)
- `zero_shot_main` costs **$1,034 USD/year**, runs **8× slower** (12,472 ms p95), and achieves an abysmal record accuracy of **0.0500** (McNemar $p < 0.0001$).
- **Business takeaway:** Following naive intuition would have cost the business more money and degraded customer experience with 12-second latency for an extractor that fails 95% of the time. The evaluation harness saved the project.

### Change-My-Mind Condition
We would only consider deploying a cascade or a larger model tier if:
1. The prompt's urgency threshold criteria are recalibrated to eliminate the downward bias.
2. An updated configuration demonstrates a statistically significant improvement on held-out test data (McNemar $p < 0.01$).
3. The marginal cost per additional record accuracy point is under **$0.05 per 1,000 tickets**.

---

## 5. Summary of Created Deliverables

1. [`labs/lab2/variants.py`](file:///d:/AI-in-Practice-Lab-main/AI-in-Practice-Lab-main/aip-lab1/labs/lab2/variants.py): Full implementation of all 7 configurations, hand-crafted few-shot exemplars, reasoned Pydantic schema, and dual-temperature cascade.
2. [`reports/lab2_grid.json`](file:///d:/AI-in-Practice-Lab-main/AI-in-Practice-Lab-main/aip-lab1/reports/lab2_grid.json): Comprehensive raw benchmark data across all 60 dev tickets and all 7 configurations.
3. [`report.md`](file:///d:/AI-in-Practice-Lab-main/AI-in-Practice-Lab-main/aip-lab1/report.md) & [`labs/lab2/report.md`](file:///d:/AI-in-Practice-Lab-main/AI-in-Practice-Lab-main/aip-lab1/labs/lab2/report.md): Formal 3-page lab report with grid tables, cascade metrics, paired tests, error clusters, confusion matrix, and negative results.
4. [`LAB2_SUMMARY.md`](file:///d:/AI-in-Practice-Lab-main/AI-in-Practice-Lab-main/aip-lab1/LAB2_SUMMARY.md): This comprehensive pedagogical and technical guide.
