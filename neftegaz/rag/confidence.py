"""Confidence on the chunk: how this fragment was read, carried to the citation.

★СТАТУС СТОИТ НА УЗЛЕ, А НЕ НА ДОКУМЕНТЕ. Документ целиком почти никогда не
бывает «хорошим» или «плохим»: в отчёте EIA STEO три четверти страниц читаются
двумя путями одинаково, а на полусотне читатели расходятся по цифрам. Статус,
приписанный документу, у первой же страницы становится ложью в одну из двух
сторон — либо порочит чистые страницы, либо покрывает грязные. Поэтому мера
приписывается фрагменту, который цитируется, и едет с ним до самой цитаты.

Ступени. Их четыре, и различие между ними — в том, ЧЕМ подтверждено чтение:

    DIRECT     два независимых пути прочли страницу одинаково
    GEOMETRY   цифры те же, но расходится их порядок или членение — значит
               текст собран по геометрии, и привязка чисел к заголовкам
               держится на нашей сборке, а не на согласии двух путей
    DISPUTED   пути расходятся ПО ЦИФРАМ: как минимум один читает неверно,
               и какой именно — сверка не решает
    UNCHECKED  сверка не выполнялась

★UNCHECKED — ОТДЕЛЬНАЯ СТУПЕНЬ, А НЕ СИНОНИМ DIRECT. Это то же правило, что в
приёмке: невыполненная проверка обязана называться невыполненной. Если бы
непроверенное молча приравнивалось к чистому, отключение сверки повысило бы
качество всех цитат разом — верный признак, что мера меряет не то.

★ОГОВОРКИ ДОКУМЕНТА ЕДУТ ВМЕСТЕ СО СТУПЕНЬЮ, НО НЕ ПОНИЖАЮТ ЕЁ. Шрифты без
ToUnicode — свойство файла, а не страницы: они означают риск неверного
декодирования везде, и одинаково на согласных и расходящихся страницах.
Понижать ими ступень значило бы смешать «этот фрагмент прочтён спорно» с «весь
документ несёт известный риск» — разные утверждения с разными последствиями.

Фрагмент, перешедший границу страниц, получает ХУДШУЮ ступень из покрытых: у
цитаты одна метка, и она должна отвечать за весь приведённый текст.
"""

from __future__ import annotations

from neftegaz.rag.crosscheck import (
    AGREE,
    DIVERGE,
    ORDER,
    TOKENIZE,
    CrossCheckReport,
    PageDiff,
)
from neftegaz.rag.crosscheck import _worst as worst_verdict
from neftegaz.rag.intake import IntakeReport

__all__ = [
    "DIRECT",
    "GEOMETRY",
    "DISPUTED",
    "UNCHECKED",
    "LEVEL_ORDER",
    "LEVEL_LABEL",
    "INDEXED_READER",
    "level_of_verdict",
    "worst_level",
    "document_caveats",
    "effective_verdict",
    "chunk_confidence",
    "annotate_chunks",
]

# Чей текст лежит в индексе. Именно о нём говорит ступень уверенности, и
# поэтому имя читателя обязано быть здесь названо: без него правило «расхождение
# сведено к чужому пути» не имеет смысла — «чужой» относительно кого?
INDEXED_READER = "pdf2xml"

DIRECT = "direct"
GEOMETRY = "geometry"
DISPUTED = "disputed"
UNCHECKED = "unchecked"

# Порядок ступеней задан явно: «хуже» — это дальше по списку, а не больше по
# алфавиту. UNCHECKED стоит между собранным и спорным намеренно: незнание хуже
# известной пересборки и лучше известного расхождения.
LEVEL_ORDER = [DIRECT, GEOMETRY, UNCHECKED, DISPUTED]

LEVEL_LABEL = {
    DIRECT: "прочитано напрямую",
    GEOMETRY: "собрано по геометрии",
    DISPUTED: "читатели расходятся",
    UNCHECKED: "сверка не выполнялась",
}

_VERDICT_LEVEL = {
    AGREE: DIRECT,
    ORDER: GEOMETRY,
    TOKENIZE: GEOMETRY,
    DIVERGE: DISPUTED,
}


def level_of_verdict(verdict: str) -> str:
    """Map one page verdict onto a confidence level."""
    return _VERDICT_LEVEL.get(verdict, UNCHECKED)


def worst_level(levels) -> str:
    """The worst of several levels; ``UNCHECKED`` when there is nothing to judge."""
    worst = None
    for level in levels:
        if worst is None or LEVEL_ORDER.index(level) > LEVEL_ORDER.index(worst):
            worst = level
    return worst if worst is not None else UNCHECKED


def effective_verdict(diff: PageDiff, indexed: str = INDEXED_READER) -> tuple[str, str | None]:
    """Вердикт, которым судят ПРОИНДЕКСИРОВАННЫЙ текст, и пояснение к нему.

    ★ЗАЧЕМ ЭТО ОТДЕЛЬНО ОТ `PageDiff.verdict`. Сводный вердикт страницы — худший
    из попарных, и это правильная запись наблюдения: где-то кто-то читает иначе.
    Но ступень уверенности отвечает на другой вопрос — можно ли верить ТОМУ
    тексту, который лежит в индексе. Это не одно и то же, и разница появилась
    ровно с третьим читателем.

    Замер по корпусу (469 страниц отчёта EIA STEO): pypdf расходится по цифрам
    на 132 страницах. Разбор десятка из них показал причину: pypdf ВИДИТ БОЛЬШЕ
    — подписи осей на графиках («-3», «0.5», «2021»…), которые poppler склеивает
    («20262026») или теряет. То есть на этих страницах наш текст сходится с
    poppler, а третий путь приносит числа из графиков, которых в нашем тексте
    нет. Понижать за это ступень — значит объявить спорными три четверти
    страниц с графиками, хотя разобранные нами числа никто не оспорил.

    Правило, которое из этого следует:

      · расхождение СВЕДЕНО к одному пути, и это НЕ наш путь → страницу судят
        оставшиеся, а расхождение выезжает оговоркой с ИМЕНЕМ пути;
      · выброс — НАШ путь → ступень понижается по худшему: когда именно наш
        читатель разошёлся с двумя остальными, это и есть повод не верить;
      · виновника нет (все разошлись вразнобой) → худший, как и было.

    Возвращает пару «вердикт, оговорка», где оговорка равна ``None``, если
    приводить нечего.
    """
    if not diff.pairs or not diff.outlier or diff.outlier == indexed:
        return diff.verdict, None

    kept = [verdict for name, verdict in diff.pairs.items() if diff.outlier not in name.split("|")]
    if not kept:
        return diff.verdict, None

    note = (
        f"с. {diff.page}: путь {diff.outlier} читает страницу иначе, "
        f"чем два остальных; текст в индексе с ними согласен"
    )
    return worst_verdict(kept), note


def document_caveats(intake: IntakeReport | None) -> list[str]:
    """Findings that belong to the file as a whole, phrased for a reader."""
    if intake is None:
        return []
    caveats = []
    for finding in intake.findings:
        if finding.code == "fonts_without_tounicode":
            caveats.append("шрифты без ToUnicode: возможно неверное декодирование символов")
        elif finding.code == "incremental_updates":
            caveats.append("файл содержит инкрементальные ревизии")
        elif finding.severity in ("warn", "broken"):
            # Всё серьёзное проносится дословно: перечислять коды по одному
            # значило бы молча терять те, которых мы ещё не встречали.
            caveats.append(finding.detail)
    return caveats


def chunk_confidence(
    page_start: int,
    page_end: int,
    page_verdicts: dict[int, str] | None,
    page_notes: dict[int, str] | None = None,
) -> tuple[str, list[str]]:
    """Level and reasons for a fragment spanning ``page_start``…``page_end``."""
    if not page_verdicts:
        return UNCHECKED, []

    levels, reasons = [], []
    for page in range(page_start, max(page_start, page_end) + 1):
        verdict = page_verdicts.get(page)
        if verdict is None:
            levels.append(UNCHECKED)
            continue
        level = level_of_verdict(verdict)
        levels.append(level)
        note = (page_notes or {}).get(page)
        if note:
            # Оговорка едет вместе со ступенью и НЕ понижает её: см.
            # `effective_verdict`. Читатель вправе знать, что третий путь
            # прочёл страницу иначе, даже когда наш текст это не порочит.
            reasons.append(note)
        if level == GEOMETRY and verdict == ORDER:
            reasons.append(f"с. {page}: пути читают числа в разном порядке")
        elif level == GEOMETRY and verdict == TOKENIZE:
            reasons.append(f"с. {page}: пути по-разному членят числа")
        elif level == DISPUTED:
            reasons.append(f"с. {page}: пути расходятся по цифрам")

    return worst_level(levels), reasons


def annotate_chunks(
    chunks: list[dict],
    crosscheck: CrossCheckReport | None = None,
    intake: IntakeReport | None = None,
) -> list[dict]:
    """Stamp every chunk with ``confidence`` and ``caveats``. Mutates in place.

    Вызывается один раз на документ: сверка стоит одного лишнего прохода по
    файлу (замер по корпусу — 4.4 с на 469 страниц), и платить его на каждый
    фрагмент незачем.
    """
    page_verdicts = None
    page_notes: dict[int, str] = {}
    if crosscheck is not None and crosscheck.pages:
        page_verdicts = {}
        for page in crosscheck.pages:
            verdict, note = effective_verdict(page)
            page_verdicts[page.page] = verdict
            if note:
                page_notes[page.page] = note

    doc_level_caveats = document_caveats(intake)

    for chunk in chunks:
        start = int(chunk.get("page_start", chunk.get("page", 0)))
        end = int(chunk.get("page_end", start))
        level, reasons = chunk_confidence(start, end, page_verdicts, page_notes)
        chunk["confidence"] = level
        chunk["caveats"] = reasons + doc_level_caveats
    return chunks
