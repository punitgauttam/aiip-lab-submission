#!/usr/bin/env python3
"""Lab 1, Parts B and C — the extractor you actually ship.

Complete the TODOs. `run_eval.py` imports `extract_b` and `extract_c` from
here, so keep those two function names.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aip.guards import _PII_PATTERNS  # noqa: E402
from aip.llm import StructuredOutputError, structured  # noqa: E402

CATEGORIES = Literal["billing", "claims", "policy_change",
                     "technical", "complaint", "information"]


# ===========================================================================
# PART B — the schema
# ===========================================================================
class TicketRecord(BaseModel):
    # Evidence comes first because it justifies the category that follows.
    evidence: str = Field(
        max_length=200,
        description="Quote verbatim the span of the ticket that determined the category. One sentence at most."
    )

    category: CATEGORIES = Field(
        description="billing = money in such as premiums, debits, refunds, invoices, tax certificates or instalments. "
                    "claims = an actual or intended claim such as cashless, reimbursement, settlement, deduction or rejection. "
                    "policy_change = altering the contract such as adding/removing a member, upgrading, porting or changing contact details. "
                    "technical = the app, portal, OTP, login, locator or document upload is broken. "
                    "complaint = Aurora's conduct itself is the subject, such as mis-selling, being kept on hold or an ignored grievance. "
                    "information = a question with no pending transaction behind it."
    )

    urgency: int = Field(
        ge=1, le=5,
        description="Urgency 1-5. "
                    "1 = a general question answerable from product knowledge or self-service without accessing the customer's record. "
                    "2 = Aurora must look up this customer's account, take action, fix a defect, or a transaction is in flight. "
                    "3 = something has already gone wrong or is stuck and the customer is waiting. "
                    "4 = repeated failure to resolve, money or access is at risk now, or the customer explicitly threatens escalation. "
                    "5 = an emergency is in progress, a formal denial demands immediate reversal, or the customer states they are escalating to the Ombudsman. "
                    "Add one level, capped at 5, if the message states a same-day or next-morning deadline. "
                    "Judge the situation, not shouting or message length."
    )

    sentiment: Literal["angry", "frustrated", "neutral", "satisfied"] = Field(
        description="angry = hostile, shouting or threatening. "
                    "frustrated = unhappy and references a prior failure such as a repeat attempt, unanswered request, delay, or something not working. "
                    "neutral = matter-of-fact and does not reference a prior failure. "
                    "satisfied = thanks or praise."
    )

    product: Literal["bronze", "silver", "gold", "platinum", "unknown"] = Field(
        description="Use bronze, silver, gold or platinum only when that plan is explicitly named in the ticket. "
                    "Use unknown when no product is explicitly named. Never infer the product."
    )

    language: Literal["en", "hi-en"] = Field(
        description="en = English only. hi-en = English mixed with Hindi words, including transliterated Hindi in Latin script."
    )
    policy_number: str | None = Field(
        default=None,
        description="A policy number must be AUR- followed by exactly 7 digits and copied verbatim. "
                    "Use only a policy number appearing in the current live message. "
                    "Ignore policy numbers appearing only in quoted replies beginning with > or in historical quoted text. "
                    "Return null when no valid policy number appears in the live message. "
                    "Never invent or reformat one."
    )

    contains_pii: bool = Field(
        default=False,
        description="True if the ticket contains a phone number or a non-Aurora email address. "
                    "A personal name alone does not count."
    )

    needs_human_review: bool = False
    review_reason: str = ""

    @field_validator("policy_number", mode="before")
    @classmethod
    def _policy_format(cls, v: str | None) -> str | None:
        if v is None:
            return None

        v = str(v).strip()

        if v.lower() in {"", "null", "none", "n/a"}:
            return None

        if not re.fullmatch(r"AUR-\d{7}", v):
            raise ValueError("policy_number must match AUR- followed by exactly 7 digits")

        return v


SYSTEM_PROMPT = """\
You are a reliable support-ticket extraction system.

TASK:
Extract the requested structured fields from the customer's CURRENT ticket text.

RULES:
1. Follow the field descriptions and schema exactly.
2. Use only information supported by the ticket.
3. Never invent, guess, or reformat information.
4. Treat quoted replies beginning with ">" as historical context, not the current message.
5. Return only the structured record required by the schema.
6. If information is absent, use the schema's permitted value such as null or unknown.
7. Evidence must be copied verbatim from the ticket.

The ticket text follows:
"""

def extract_b(ticket: str) -> TicketRecord:
    try:
        return structured(
            ticket,
            schema=TicketRecord,
            system=SYSTEM_PROMPT,
            tier="SMALL",
        )
    except StructuredOutputError as e:
        return TicketRecord(
            evidence="",
            category="information",
            urgency=1,
            sentiment="neutral",
            product="unknown",
            language="en",
            policy_number=None,
            contains_pii=False,
            needs_human_review=True,
            review_reason=str(e),
        )
    except Exception as e:
        return TicketRecord(
            evidence="",
            category="information",
            urgency=1,
            sentiment="neutral",
            product="unknown",
            language="en",
            policy_number=None,
            contains_pii=False,
            needs_human_review=True,
            review_reason=str(e),
        )
        


# ===========================================================================
# PART C — move the deterministic work out of the model
# ===========================================================================
POLICY_RE = re.compile(r"\bAUR-\d{7}\b")

# The quoted-reply marker. Everything after this is history, not the current
# message. Part C3 asks you to decide what that means for policy extraction.
QUOTE_MARKER = re.compile(r"^\s*>", re.MULTILINE)


def extract_deterministic(ticket: str) -> dict:
    # Only use policy numbers from the live message.
    # Anything after a quoted-reply line (starting with '>') is treated as history.
    live_text = QUOTE_MARKER.split(ticket, maxsplit=1)[0]

    policy_match = POLICY_RE.search(live_text)
    policy_number = policy_match.group(0) if policy_match else None

    contains_pii = any(
        pattern.search(ticket)
        for pattern in _PII_PATTERNS.values()
    )

    return {
        "policy_number": policy_number,
        "contains_pii": contains_pii,
    }


def apply_business_rules(rec_fields: dict, ticket: str) -> dict:
    rec_fields["escalate"] = (
        rec_fields["urgency"] >= 4
        or "ombudsman" in ticket.lower()
    )
    return rec_fields


class TicketRecordC(BaseModel):
    evidence: str = Field(
        max_length=200,
        description="Quote verbatim the span of the ticket that determined the category. One sentence at most."
    )

    category: CATEGORIES = Field(
        description="billing = money in such as premiums, debits, refunds, invoices, tax certificates or instalments. "
                    "claims = an actual or intended claim such as cashless, reimbursement, settlement, deduction or rejection. "
                    "policy_change = altering the contract such as adding/removing a member, upgrading, porting or changing contact details. "
                    "technical = the app, portal, OTP, login, locator or document upload is broken. "
                    "complaint = Aurora's conduct itself is the subject, such as mis-selling, being kept on hold or an ignored grievance. "
                    "information = a question with no pending transaction behind it."
    )

    urgency: int = Field(
        ge=1,
        le=5,
        description=(
            "Urgency 1-5. "
            "1 = a general product question or self-service how-to that can be answered "
            "without opening the customer's record. "
            "2 = Aurora must look up this customer's account, take an action, fix a defect, "
            "or a transaction is currently in progress. "
            "3 = something has already gone wrong or is stuck and the customer is waiting. "
            "4 = repeated failure to resolve, money or access is at risk now, or the customer "
            "threatens escalation. "
            "5 = an emergency is in progress, a formal denial demands immediate reversal, "
            "or the customer states that they are filing/escalating to the Ombudsman. "
            "Add 1 for a same-day or next-morning deadline, capped at 5. "
            "Do not use anger or message length to determine urgency. "
            "A threat to go to the Ombudsman is 4; stating that the customer is filing "
            "with the Ombudsman is 5."
        )
    )

    sentiment: Literal[
        "angry", "frustrated", "neutral", "satisfied"
    ] = Field(
        description="angry = hostile, shouting or threatening. "
                    "frustrated = unhappy because of a prior failure, repeat attempt, unanswered request or delay. "
                    "neutral = matter-of-fact with no prior failure. "
                    "satisfied = thanks or praise."
    )

    product: Literal[
        "bronze", "silver", "gold", "platinum", "unknown"
    ] = Field(
        description="Use bronze, silver, gold or platinum only when that plan is explicitly named in the ticket. "
                    "Use unknown when no product is explicitly named. Never infer the product."
    )

    language: Literal["en", "hi-en"] = Field(
        description="en = English only. hi-en = English mixed with Hindi words, including transliterated Hindi in Latin script."
    )


def extract_c(ticket: str) -> dict:
    try:
        rec = structured(
            ticket,
            schema=TicketRecordC,
            system=SYSTEM_PROMPT,
            tier="SMALL",
        )

        result = rec.model_dump()

        result.update(extract_deterministic(ticket))
        result = apply_business_rules(result, ticket)

        return result

    except StructuredOutputError as e:
        result = {
            "evidence": "",
            "category": "information",
            "urgency": 1,
            "sentiment": "neutral",
            "product": "unknown",
            "language": "en",
            "needs_human_review": True,
            "review_reason": str(e),
        }

        result.update(extract_deterministic(ticket))
        result = apply_business_rules(result, ticket)

        return result


if __name__ == "__main__":
    import json

    root = Path(__file__).resolve().parents[2]
    sample = json.loads(
        (root / "data/eval/extraction_dev.jsonl").open(encoding="utf-8").readline()
    )
    print("--- ticket ---")
    print(sample["input"][:600])
    print("\n--- gold ---")
    print(sample["expected"])
    print("\n--- yours ---")
    print(extract_c(sample["input"]))