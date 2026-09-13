#!/usr/bin/env python3
"""Lab 2 — the configurations under test.

Each variant is a callable `str -> dict`. `grid.py` runs them all through the
same harness, so the only thing that differs between rows of your table is the
thing you intended to differ.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pydantic import Field  # noqa: E402

from aip.llm import StructuredOutputError, structured  # noqa: E402
from labs.lab1.extract import (  # noqa: E402
    SYSTEM_PROMPT, TicketRecord, apply_business_rules, extract_deterministic,
)

# ---------------------------------------------------------------------------
# A1 — your six chosen examples.
# ---------------------------------------------------------------------------
# TODO A1: choose 6 dev-set tickets. For EACH, write one line saying what it
#          teaches that prose cannot. Pick edges, not averages (T2 §2.2):
#            - the billing/complaint boundary
#            - a ticket with no policy number (teaches null)
#            - a Hinglish ticket
#            - a satisfied-but-urgent ticket (the sentiment/urgency trap)
#            - a ticket whose policy number is only in a quoted reply
#            - one you got wrong in Lab 1
FEW_SHOT_IDS: list[str] = [
    "T0054",  # teaches: mis-selling complaints with refund demands are 'complaint', not 'billing' (misconduct overrides 'refund')
    "T0048",  # teaches: 'my policy' without an AUR- identifier must return policy_number=null and product='unknown'
    "T0200",  # teaches: transliterated Hindi ('Koi solution batayiye') requires language='hi-en' and prior deductions mean frustrated
    "T0029",  # teaches: polite expressions of gratitude ('Thanks for...') indicate sentiment='satisfied' despite claim mention
    "T0238",  # teaches: reference numbers in quoted history (SR-100238) are not policy numbers; policy_number must be null
    "T0021",  # teaches: next-morning deadline at hospital insurance desk sets urgency=5, angry sentiment, and language='hi-en'
]

FEW_SHOT_EVIDENCE: dict[str, str] = {
    "T0054": "Your agent mis-sold me this policy.",
    "T0048": "Please add my mother as a dependent on my policy.",
    "T0200": "My claim on AUR-9674338 was settled at Rs 41800 but the hospital bill was much higher.",
    "T0029": "Thanks for settling my claim on my policy so quickly.",
    "T0238": "I submitted a portability request 21 days ago and heard nothing.",
    "T0021": "Your portal has been down all morning and I am standing at the hospital insurance desk trying to show my Aurora Gold policy my policy.",
}

FEW_SHOT_REASONING: dict[str, str] = {
    "T0054": "The customer is complaining that an agent mis-sold the policy regarding maternity waiting periods. Although a refund is requested, the primary subject is company/agent misconduct, so category is complaint. Urgency is 4 due to explicit demand/grievance.",
    "T0048": "Customer requests adding mother as dependent on 'my policy'. No AUR- number is present, so policy_number is null and product is unknown. Category is policy_change, urgency is 2.",
    "T0200": "Customer asks about proportionate deduction on a settled claim. The phrase 'Koi solution batayiye' is Hindi, so language is hi-en. Prior deduction issue reflects frustrated sentiment, urgency 3, category claims.",
    "T0029": "Customer expresses gratitude for prompt settlement and asks an informational question about NCB impact. Tone is satisfied, urgency 1, category information.",
    "T0238": "Customer follows up on a portability request. Reference SR-100238 in the quoted support reply is a service request, not a policy number. Live text names Bronze plan. Category policy_change, urgency 3, sentiment frustrated.",
    "T0021": "Customer at hospital insurance desk cannot access portal, with admission deadline tomorrow morning. Urgency is 5 due to immediate hospital deadline, sentiment angry, category technical, language hi-en ('Jaldi karo please').",
}


def load_examples(ids: list[str]) -> list[dict]:
    rows = [json.loads(l) for l in
            (ROOT / "data/eval/extraction_dev.jsonl").open(encoding="utf-8")]
    by_id = {r["id"]: r for r in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown example ids: {missing}")
    return [by_id[i] for i in ids]


def few_shot_block(ids: list[str], include_reasoning: bool = False) -> str:
    """Render the examples into the prompt.

    The example output format is byte-identical to the format we ask the model
    to produce.
    """
    examples = load_examples(ids)
    blocks = []
    for ex in examples:
        tid = ex["id"]
        exp = ex["expected"]
        evidence = FEW_SHOT_EVIDENCE.get(tid, "")

        record = {}
        if include_reasoning:
            record["reasoning"] = FEW_SHOT_REASONING.get(tid, "")
        record["evidence"] = evidence
        record["category"] = exp["category"]
        record["urgency"] = exp["urgency"]
        record["sentiment"] = exp["sentiment"]
        record["product"] = exp["product"]
        record["language"] = exp["language"]
        record["policy_number"] = exp["policy_number"]
        record["contains_pii"] = exp["contains_pii"]

        block = (
            f"--- EXAMPLE TICKET ---\n"
            f"{ex['input'].strip()}\n\n"
            f"EXPECTED JSON OUTPUT:\n"
            f"{json.dumps(record, indent=2)}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# The variants
# ---------------------------------------------------------------------------
def zero_shot(ticket: str, tier: str = "SMALL") -> dict:
    """Lab 1 Part C baseline, no examples."""
    try:
        rec = structured(
            ticket,
            schema=TicketRecord,
            system=SYSTEM_PROMPT,
            tier=tier,
        )
        result = rec.model_dump()
    except (StructuredOutputError, Exception) as e:
        result = {
            "evidence": "",
            "category": "information",
            "urgency": 1,
            "sentiment": "neutral",
            "product": "unknown",
            "language": "en",
            "policy_number": None,
            "contains_pii": False,
            "needs_human_review": True,
            "review_reason": str(e),
        }

    result.update(extract_deterministic(ticket))
    result = apply_business_rules(result, ticket)
    return result


def few_shot(ticket: str, tier: str = "SMALL") -> dict:
    """zero_shot + the few-shot block."""
    examples_str = few_shot_block(FEW_SHOT_IDS, include_reasoning=False)
    sys_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Here are labelled examples demonstrating the classification and extraction rules:\n\n"
        f"{examples_str}"
    )
    try:
        rec = structured(
            ticket,
            schema=TicketRecord,
            system=sys_prompt,
            tier=tier,
        )
        result = rec.model_dump()
    except (StructuredOutputError, Exception) as e:
        result = {
            "evidence": "",
            "category": "information",
            "urgency": 1,
            "sentiment": "neutral",
            "product": "unknown",
            "language": "en",
            "policy_number": None,
            "contains_pii": False,
            "needs_human_review": True,
            "review_reason": str(e),
        }

    result.update(extract_deterministic(ticket))
    result = apply_business_rules(result, ticket)
    return result


class TicketRecordReasoned(TicketRecord):
    """Add a `reasoning: str` field FIRST (T2 §3.3).

    Pydantic keeps declaration order, and field order in the JSON Schema
    influences generation order. Putting reasoning first makes it condition the
    answer; putting it last makes it a post-hoc rationalisation.
    """
    reasoning: str = Field(
        default="",
        description="Step-by-step reasoning explaining which category applies and why, based on the customer situation."
    )

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        s = super().model_json_schema(*args, **kwargs)
        props = s.get("properties", {})
        if "reasoning" in props:
            new_props = {"reasoning": props["reasoning"]}
            for k, v in props.items():
                if k != "reasoning":
                    new_props[k] = v
            s["properties"] = new_props
        reqs = s.get("required", [])
        if "reasoning" in reqs:
            s["required"] = ["reasoning", *[r for r in reqs if r != "reasoning"]]
        return s


def few_shot_reasoned(ticket: str, tier: str = "SMALL") -> dict:
    """few_shot with TicketRecordReasoned."""
    examples_str = few_shot_block(FEW_SHOT_IDS, include_reasoning=True)
    sys_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Here are labelled examples demonstrating the reasoning and extraction rules. "
        "Provide your reasoning in the reasoning field first, before the other fields:\n\n"
        f"{examples_str}"
    )
    try:
        rec = structured(
            ticket,
            schema=TicketRecordReasoned,
            system=sys_prompt,
            tier=tier,
        )
        result = rec.model_dump()
    except (StructuredOutputError, Exception) as e:
        result = {
            "reasoning": "",
            "evidence": "",
            "category": "information",
            "urgency": 1,
            "sentiment": "neutral",
            "product": "unknown",
            "language": "en",
            "policy_number": None,
            "contains_pii": False,
            "needs_human_review": True,
            "review_reason": str(e),
        }

    result.update(extract_deterministic(ticket))
    result = apply_business_rules(result, ticket)
    return result


def cascade(ticket: str) -> dict:
    """SMALL first; escalate to MAIN on validation failure, empty evidence, or two-sample disagreement.

    Record which path each ticket took -- set rec['_path'] = 'small' | 'large'
    so grid.py can report the escalation rate.
    """
    escalate_trigger = False

    # Sample 1: SMALL model at temperature=0.0
    try:
        rec1 = structured(
            ticket,
            schema=TicketRecord,
            system=SYSTEM_PROMPT,
            tier="SMALL",
            temperature=0.0,
        )
        dict1 = rec1.model_dump()
    except Exception:
        dict1 = None
        escalate_trigger = True

    if dict1 is not None:
        ev = dict1.get("evidence", "").strip()
        if len(ev) < 5:
            escalate_trigger = True
        else:
            # Sample 2: SMALL model at temperature=0.7 to check self-consistency.
            # Using temperature > 0 changes both sampling and cache key.
            try:
                rec2 = structured(
                    ticket,
                    schema=TicketRecord,
                    system=SYSTEM_PROMPT,
                    tier="SMALL",
                    temperature=0.7,
                )
                dict2 = rec2.model_dump()
                if (dict1.get("category") != dict2.get("category")
                        or dict1.get("urgency") != dict2.get("urgency")
                        or dict1.get("sentiment") != dict2.get("sentiment")):
                    escalate_trigger = True
            except Exception:
                escalate_trigger = True

    if escalate_trigger:
        try:
            rec_main = structured(
                ticket,
                schema=TicketRecord,
                system=SYSTEM_PROMPT,
                tier="MAIN",
                temperature=0.0,
            )
            result = rec_main.model_dump()
        except Exception as e:
            result = dict1 or {
                "evidence": "",
                "category": "information",
                "urgency": 1,
                "sentiment": "neutral",
                "product": "unknown",
                "language": "en",
                "policy_number": None,
                "contains_pii": False,
                "needs_human_review": True,
                "review_reason": str(e),
            }
        result["_path"] = "large"
    else:
        result = dict1
        result["_path"] = "small"

    result.update(extract_deterministic(ticket))
    result = apply_business_rules(result, ticket)
    return result


VARIANTS = {
    "zero_shot": lambda t: zero_shot(t, "SMALL"),
    "zero_shot_main": lambda t: zero_shot(t, "MAIN"),
    "few_shot": lambda t: few_shot(t, "SMALL"),
    "few_shot_main": lambda t: few_shot(t, "MAIN"),
    "few_shot_reasoned": lambda t: few_shot_reasoned(t, "SMALL"),
    "few_shot_reasoned_main": lambda t: few_shot_reasoned(t, "MAIN"),
    "cascade": cascade,
}
