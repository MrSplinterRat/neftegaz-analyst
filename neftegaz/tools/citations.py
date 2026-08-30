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

__all__ = ["REQUIRED_FIELDS", "CONFIDENCE_MARK", "format_claim", "format_answer"]

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "report": ("text", "source_name", "date", "page"),
    "web": ("text", "source_name"),
}

# ★КАК ФРАГМЕНТ БЫЛ ПРОЧИТАН — ЧАСТЬ ССЫЛКИ, А НЕ ПРИМЕЧАНИЕ К НЕЙ.
# Ссылка обещает проверяемость: «откройте страницу 14 и увидите то же самое».
# Обещание держится ровно настолько, насколько мы уверены, что прочли эту
# страницу верно. Если два независимых пути чтения дали на ней разные цифры,
# умолчать об этом — значит выдать спорное за проверенное; ровно то, что
# докстринг этого модуля называет самым вредным, что он мог бы сделать.
#
# «Прочитано напрямую» не печатается: у чистой ссылки метка должна оставаться
# чистой, иначе оговорка на спорной перестанет бросаться в глаза.
CONFIDENCE_MARK: dict[str, str] = {
    "direct": "",
    "geometry": "текст собран по геометрии страницы",
    "disputed": "⚠ два пути чтения расходятся по цифрам",
    "unchecked": "сверка чтения не выполнялась",
}


def format_claim(claim: dict) -> str:
    """Render one claim with its citation appended."""
    source_type = claim.get("source_type")
    if source_type not in REQUIRED_FIELDS:
        raise ValueError(
            f"unknown source_type: {source_type!r}; expected one of {sorted(REQUIRED_FIELDS)}"
        )
    for field in REQUIRED_FIELDS[source_type]:
        if field not in claim:
            raise KeyError(field)

    if source_type == "report":
        inside = f"Отчёт {claim['source_name']}, {claim['date']}, с. {claim['page']}"
        # Отсутствие поля — это «не проверяли», а не «проверено и чисто».
        # Умолчание в другую сторону сделало бы отключение сверки способом
        # улучшить все цитаты разом.
        note = CONFIDENCE_MARK.get(claim.get("confidence", "unchecked"), "")
        if note:
            inside = f"{inside}; {note}"
        mark = f"[{inside}]"
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
