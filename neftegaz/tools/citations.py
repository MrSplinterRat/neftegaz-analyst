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

import re
from typing import Any

from neftegaz.rag.confidence import DIRECT, LEVEL_ORDER

__all__ = [
    "REQUIRED_FIELDS",
    "CONFIDENCE_MARK",
    "CONFIDENCE_ADVICE",
    "CITATION",
    "format_claim",
    "format_answer",
    "annotate_answer",
]

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

# ★ЧТО ЧИТАТЕЛЮ С ЭТИМ ДЕЛАТЬ. Метка внутри скобок коротка намеренно — она стои́т
# посреди фразы и не должна её разрывать, — но короткая метка называет причину и
# молчит о последствии. «Два пути чтения расходятся по цифрам» понятно нам и
# ничего не говорит человеку, который держит в руках число и решает, ставить ли
# на него деньги. Поэтому у каждой ступени есть вторая, длинная форма: она
# печатается ОДИН раз в конце ответа и отвечает ровно на вопрос «и что теперь».
CONFIDENCE_ADVICE: dict[str, str] = {
    "geometry": (
        "цифры на странице прочитаны одинаково, а вот к какому заголовку и столбцу "
        "относится каждая — это уже наша сборка страницы, а не согласие двух путей "
        "чтения. Сверь по отчёту не само число, а его место в таблице: год, квартал, "
        "единицы."
    ),
    "disputed": (
        "две независимые программы прочли эту страницу по-разному, и сверка не решает, "
        "какая из них права. Значит, число в ответе может быть верным, а может быть "
        "чужим из соседней строки. Прежде чем опираться на него в решении, открой "
        "страницу отчёта и прочти её глазами."
    ),
    "unchecked": (
        "эту страницу читал один путь, вторым её никто не перечитывал. Это не значит, "
        "что она прочтена неверно, — это значит, что мы не проверяли. Если число "
        "существенно для решения, сверь его со страницей."
    ),
}

# «[Отчёт EIA STEO, июль 2026, с. 34–35]» — и вариант с одной страницей.
#
# ★ШАБЛОН ЖИВЁТ ЗДЕСЬ, В ОДНОМ ЭКЗЕМПЛЯРЕ. Его читают двое: этот модуль, который
# дописывает ступень в готовый ответ, и внешняя сверка цитат, которая проверяет,
# стои́т ли названное число на названной странице. Две копии одного шаблона
# разошлись бы при первой же правке формата ссылки, и разошлись бы МОЛЧА: сверка
# просто перестала бы видеть цитаты и отчиталась бы о чистоте.
#
# ★ПОМЕТКА О ЧТЕНИИ ВХОДИТ В ШАБЛОН ПЯТОЙ ГРУППОЙ, А НЕ ЛОМАЕТ ЕГО. Первая
# редакция требовала «]» сразу за номерами страниц — и тогда ссылка, которой
# дописали ступень, переставала быть ссылкой для всех прочих читателей шаблона:
# сверка цитат молча перестала бы видеть ровно те цитаты, о которых мы честно
# сказали, что они спорные. Худший из возможных исходов: чем громче оговорка,
# тем меньше проверки.
CITATION = re.compile(
    r"\[Отчёт\s+([^,\]]+),\s*([а-яё]+\s+\d{4}),\s*с\.\s*(\d+)"
    r"(?:\s*[–—-]\s*(\d+))?(;\s*[^\]]*)?\]"
)


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


def _span(first: int, last: int) -> str:
    return f"с. {first}" if last == first else f"с. {first}–{last}"


def annotate_answer(answer: str, hits: list[Any]) -> tuple[str, dict]:
    """Дописать ступень чтения в ссылки готового ответа.

    ★ПОМЕТКУ СТАВИТ КОД, А НЕ МОДЕЛЬ. Ступень посчитана при индексации и доезжает
    до ответа в поле фрагмента; модели она не показывается вовсе. Показать
    значило бы поручить пересказчику воспроизвести оговорку — а пересказчик
    вправе счесть её неважной и потерять, причём молча и не всегда. Ровно этим
    соображением оговорка про молчащие источники уже отдана коду.

    ★СТУПЕНЬ БЕРЁТСЯ У ТЕХ ФРАГМЕНТОВ, КОТОРЫЕ РЕАЛЬНО ПОДАНЫ МОДЕЛИ. Ссылка на
    страницу, которой в контексте не было, ступени не получает: сказать о ней
    нечего, а поставить «прочитано напрямую» по умолчанию — это заверить
    непроверенное. Такие ссылки считаются отдельным числом, потому что их
    появление означает, что модель сослалась на то, чего не читала, и об этом
    должна знать сверка цитат, а не только читатель.

    ★ССЫЛКА, НАКРЫВАЮЩАЯ НЕСКОЛЬКО ФРАГМЕНТОВ, ПОЛУЧАЕТ ХУДШУЮ ИЗ ИХ СТУПЕНЕЙ —
    то же правило, по которому ступень получает фрагмент, перешедший границу
    страниц: метка одна, и она отвечает за весь приведённый по ссылке текст.

    Возвращает размеченный текст и замер: сколько ссылок на отчёт было, скольким
    приписана пометка, сколько не совпало ни с одним поданным фрагментом.
    """
    marked: dict[str, set[str]] = {}
    stats = {"citations": 0, "marked": 0, "unmatched": 0}

    def mark(found: re.Match) -> str:
        source_name, date, first, last, existing = found.groups()
        first_page = int(first)
        last_page = int(last) if last else first_page
        stats["citations"] += 1
        # Ссылка, уже несущая пометку, второй не получает: разметка идемпотентна,
        # и повторный проход по готовому ответу его не портит.
        if existing:
            return found.group(0)
        levels = [
            hit.confidence
            for hit in hits
            if hit.source_name == source_name
            and hit.date == date
            and not (hit.page_end < first_page or hit.page > last_page)
        ]
        if not levels:
            stats["unmatched"] += 1
            return found.group(0)
        worst = max(levels, key=LEVEL_ORDER.index)
        note = CONFIDENCE_MARK.get(worst, "")
        if worst == DIRECT or not note:
            return found.group(0)
        stats["marked"] += 1
        marked.setdefault(worst, set()).add(
            f"{source_name}, {date}, {_span(first_page, last_page)}"
        )
        return f"{found.group(0)[:-1]}; {note}]"

    text = CITATION.sub(mark, answer)
    if not marked:
        return text, stats

    lines = [
        "\n\n---\n",
        "**Как читать пометки у ссылок.** Ступень стои́т на странице отчёта и говорит, "
        "насколько уверенно мы прочли именно её. Ссылки без пометки прочитаны двумя "
        "независимыми путями одинаково.\n",
    ]
    # Худшее — первым: читатель, дочитавший до этого места одну строку, должен
    # прочесть ту, которая может стоить ему решения.
    for level in reversed(LEVEL_ORDER):
        if level not in marked:
            continue
        pages = "; ".join(sorted(marked[level]))
        lines.append(f"* **{CONFIDENCE_MARK[level]}** ({pages}) — {CONFIDENCE_ADVICE[level]}")
    return text + "\n".join(lines), stats
