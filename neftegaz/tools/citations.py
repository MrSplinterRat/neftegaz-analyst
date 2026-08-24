"""Source marking.

Requirement 2.4 of the assignment: the answer must state, per claim, where the
information came from. Two shapes exist and they are deliberately not
interchangeable —

    [Отчёт OPEC MOMR, март 2025, с. 14]
    [Источник: Reuters, web]

A verified report citation carries a date and a page so a reader can open the
document and check. A web citation cannot promise that, and pretending
otherwise — by inventing a page or padding the label — would be the single most
damaging thing this module could do: it would make an unverified claim look
verified. So a missing field raises instead of defaulting.
"""

from __future__ import annotations

__all__ = ["REQUIRED_FIELDS", "format_claim", "format_answer"]

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "report": ("text", "source_name", "date", "page"),
    "web": ("text", "source_name"),
}


def format_claim(claim: dict) -> str:
    """Render one claim with its citation appended."""
    source_type = claim.get("source_type")
    if source_type not in REQUIRED_FIELDS:
        raise ValueError(
            f"unknown source_type: {source_type!r}; "
            f"expected one of {sorted(REQUIRED_FIELDS)}"
        )
    for field in REQUIRED_FIELDS[source_type]:
        if field not in claim:
            raise KeyError(field)

    if source_type == "report":
        mark = f"[Отчёт {claim['source_name']}, {claim['date']}, с. {claim['page']}]"
    else:
        mark = f"[Источник: {claim['source_name']}, web]"
    return f"{claim['text']} {mark}"


def format_answer(claims: list[dict]) -> str:
    """Render a list of claims as paragraphs, each with its own citation.

    Paragraph-level rather than answer-level marking is what makes a *combined*
    answer honest: when part of a reply rests on a report and part on the web,
    a single trailing citation would silently lend the report's authority to
    the web-sourced half.
    """
    return "\n\n".join(format_claim(claim) for claim in claims)
