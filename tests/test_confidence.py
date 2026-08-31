"""Tests for the confidence level carried on each chunk.

These check the property that makes the level worth having: it can only ever
overstate a fragment by mistake, never by design. Nothing silently becomes
«read directly» — not a missing verdict, not a missing cross-check, not an old
point in the collection indexed before the check existed.
"""

from __future__ import annotations

from neftegaz.rag.confidence import (
    DIRECT,
    DISPUTED,
    GEOMETRY,
    UNCHECKED,
    annotate_chunks,
    chunk_confidence,
    document_caveats,
    level_of_verdict,
    worst_level,
)
from neftegaz.rag.crosscheck import AGREE, DIVERGE, ORDER, TOKENIZE, CrossCheckReport, compare_pages
from neftegaz.rag.intake import inspect_pdf

# ── отображение вердиктов ──────────────────────────────────────────────────


def test_each_verdict_maps_to_its_level():
    assert level_of_verdict(AGREE) == DIRECT
    assert level_of_verdict(ORDER) == GEOMETRY
    assert level_of_verdict(TOKENIZE) == GEOMETRY
    assert level_of_verdict(DIVERGE) == DISPUTED


def test_an_unknown_verdict_is_unchecked_not_direct():
    """A verdict we do not recognise must not be read as good news."""
    assert level_of_verdict("something-new") == UNCHECKED


def test_worst_level_wins_and_an_empty_list_is_unchecked():
    assert worst_level([DIRECT, GEOMETRY]) == GEOMETRY
    assert worst_level([DIRECT, DISPUTED, GEOMETRY]) == DISPUTED
    assert worst_level([DIRECT, UNCHECKED]) == UNCHECKED
    assert worst_level([]) == UNCHECKED


# ── фрагмент ───────────────────────────────────────────────────────────────


def test_a_chunk_on_one_clean_page_is_direct():
    level, reasons = chunk_confidence(3, 3, {3: AGREE})
    assert level == DIRECT
    assert reasons == []


def test_a_chunk_spanning_a_clean_and_a_contested_page_takes_the_worse():
    """One citation, one mark — and it must answer for the whole quoted text."""
    level, reasons = chunk_confidence(3, 4, {3: AGREE, 4: DIVERGE})
    assert level == DISPUTED
    assert reasons == ["с. 4: пути расходятся по цифрам"]


def test_reasons_name_the_page_and_the_kind_of_disagreement():
    _level, reasons = chunk_confidence(1, 2, {1: ORDER, 2: TOKENIZE})
    assert reasons == [
        "с. 1: пути читают числа в разном порядке",
        "с. 2: пути по-разному членят числа",
    ]


def test_a_page_missing_from_the_cross_check_is_unchecked():
    level, _reasons = chunk_confidence(9, 9, {1: AGREE})
    assert level == UNCHECKED


def test_no_cross_check_at_all_is_unchecked():
    assert chunk_confidence(1, 1, None) == (UNCHECKED, [])
    assert chunk_confidence(1, 1, {}) == (UNCHECKED, [])


# ── оговорки документа ─────────────────────────────────────────────────────


def test_document_caveats_are_absent_without_an_intake_report():
    assert document_caveats(None) == []


def test_font_caveat_does_not_lower_the_level_but_is_carried(tmp_path):
    """A file-wide risk is a different statement from a contested page."""
    broken = tmp_path / "x.pdf"
    broken.write_bytes(b"not a pdf at all")
    intake = inspect_pdf(str(broken))

    chunks = [{"page_start": 1, "page_end": 1, "page": 1, "text": "x"}]
    annotate_chunks(chunks, crosscheck=None, intake=intake)
    # Уровень определяется сверкой, которой не было, — значит unchecked, а
    # находки приёмки едут рядом, не подменяя его.
    assert chunks[0]["confidence"] == UNCHECKED
    assert chunks[0]["caveats"], "находки приёмки должны доехать до фрагмента"


# ── склейка всего вместе ───────────────────────────────────────────────────


def make_report(verdicts: dict[int, str]) -> CrossCheckReport:
    report = CrossCheckReport(path="x.pdf")
    for page, verdict in sorted(verdicts.items()):
        # Строим страницы через реальную функцию сравнения, чтобы тест не
        # зависел от того, как именно устроен PageDiff внутри.
        pairs = {
            AGREE: ("1", "1"),
            ORDER: ("1 2", "2 1"),
            TOKENIZE: ("11", "1 1"),
            DIVERGE: ("1", "1 2"),
        }[verdict]
        diff = compare_pages(pairs[0], pairs[1], page)
        assert diff.verdict == verdict
        report.pages.append(diff)
    return report


def test_annotate_stamps_every_chunk():
    chunks = [
        {"page_start": 1, "page_end": 1, "page": 1, "text": "a"},
        {"page_start": 2, "page_end": 3, "page": 2, "text": "b"},
    ]
    annotate_chunks(chunks, crosscheck=make_report({1: AGREE, 2: AGREE, 3: DIVERGE}))
    assert chunks[0]["confidence"] == DIRECT
    assert chunks[1]["confidence"] == DISPUTED
    assert all("caveats" in chunk for chunk in chunks)


def test_annotate_without_a_cross_check_marks_everything_unchecked():
    chunks = [{"page_start": 1, "page_end": 1, "page": 1, "text": "a"}]
    annotate_chunks(chunks)
    assert chunks[0]["confidence"] == UNCHECKED


def test_confidence_reaches_the_citation():
    """The end-to-end property: what the reader finally sees."""
    from neftegaz.rag.store import Hit
    from neftegaz.tools.citations import format_claim

    hit = Hit(
        text="13.28 13.51",
        score=0.7,
        source_name="EIA STEO",
        date="июль 2026",
        page=22,
        page_end=22,
        confidence=DISPUTED,
        caveats=("с. 22: пути расходятся по цифрам",),
    )
    rendered = format_claim(hit.as_claim())
    assert "с. 22" in rendered
    assert "расходятся" in rendered


# ── ступень доезжает до ТЕКСТА ответа, а не только до карточки источника ────
#
# ★Мера этой ветки объявлена до кода и состоит из двух половин, потому что одна
# половина без другой ничего не доказывает. Верхняя: ответ, стоящий на спорных
# страницах, обязан ВЫГЛЯДЕТЬ иначе — иначе метка декоративна. Нижняя: ответ,
# стоящий на чистых страницах, обязан остаться БАЙТ В БАЙТ прежним — иначе мы
# просто покрасили всё подряд, и пометка перестала что-либо различать.


ANSWER = (
    "Добыча в США — 13.28 млн барр./сут [Отчёт EIA STEO, июль 2026, с. 35–36]. "
    "Тот же источник называет 13.51 [Отчёт EIA STEO, июль 2026, с. 35–36]. "
    "Потребление приведено отдельно [Отчёт EIA STEO, май 2026, с. 40–41]."
)


def make_hit(date: str, page: int, page_end: int, confidence: str):
    from neftegaz.rag.store import Hit

    return Hit(
        text="13.28 13.51",
        score=0.7,
        source_name="EIA STEO",
        date=date,
        page=page,
        page_end=page_end,
        confidence=confidence,
    )


def test_clean_answer_is_left_byte_for_byte_alone():
    from neftegaz.tools.citations import annotate_answer

    hits = [
        make_hit("июль 2026", 35, 36, DIRECT),
        make_hit("май 2026", 40, 41, DIRECT),
    ]
    text, stats = annotate_answer(ANSWER, hits)
    assert text == ANSWER
    assert stats == {"citations": 3, "marked": 0, "unmatched": 0}


def test_a_disputed_answer_looks_different_and_says_what_to_do():
    from neftegaz.tools.citations import annotate_answer

    hits = [
        make_hit("июль 2026", 35, 36, DISPUTED),
        make_hit("май 2026", 40, 41, GEOMETRY),
    ]
    text, stats = annotate_answer(ANSWER, hits)
    assert text != ANSWER
    assert stats == {"citations": 3, "marked": 3, "unmatched": 0}
    # Пометка стои́т внутри ссылки, у каждой её встречи.
    assert text.count("⚠ два пути чтения расходятся по цифрам") == 3
    assert "с. 40–41; текст собран по геометрии страницы]" in text
    # И один раз — что читателю с этим делать.
    assert text.count("Как читать пометки у ссылок") == 1
    assert "открой страницу отчёта и прочти её глазами" in text
    assert "год, квартал, единицы" in text


def test_the_worst_level_of_the_covered_fragments_wins():
    """Одна ссылка — одна метка, и она отвечает за весь приведённый текст."""
    from neftegaz.tools.citations import annotate_answer

    hits = [
        make_hit("июль 2026", 35, 35, DIRECT),
        make_hit("июль 2026", 36, 36, DISPUTED),
        make_hit("май 2026", 40, 41, DIRECT),
    ]
    text, stats = annotate_answer(ANSWER, hits)
    assert stats["marked"] == 2
    assert "с. 35–36; ⚠ два пути чтения расходятся по цифрам]" in text
    assert "с. 40–41]" in text


def test_a_citation_no_fed_fragment_covers_is_counted_not_guessed():
    """«Прочитано напрямую» по умолчанию заверило бы непроверенное."""
    from neftegaz.tools.citations import annotate_answer

    text, stats = annotate_answer(ANSWER, [make_hit("июль 2026", 35, 36, DIRECT)])
    assert text == ANSWER
    assert stats["unmatched"] == 1
    assert stats["marked"] == 0


def test_marking_twice_does_not_stack_marks():
    from neftegaz.tools.citations import annotate_answer

    hits = [make_hit("июль 2026", 35, 36, DISPUTED), make_hit("май 2026", 40, 41, DIRECT)]
    once, _ = annotate_answer(ANSWER, hits)
    twice, stats = annotate_answer(once, hits)
    assert twice == once
    assert stats["marked"] == 0
    assert stats["citations"] == 3


def test_a_marked_citation_is_still_a_citation_for_the_checker():
    """★Худший исход был бы такой: чем громче оговорка, тем меньше проверки.

    Сверка цитат находит ссылки тем же шаблоном. Если бы дописанная пометка
    выводила ссылку из-под шаблона, спорные цитаты — ровно те, которые важнее
    всего проверить, — молча перестали бы проверяться, а отчёт сверки остался
    бы зелёным.
    """
    from neftegaz.tools.citations import CITATION, annotate_answer

    hits = [make_hit("июль 2026", 35, 36, DISPUTED), make_hit("май 2026", 40, 41, DIRECT)]
    text, _ = annotate_answer(ANSWER, hits)
    found = CITATION.findall(text)
    assert len(found) == 3
    assert [(row[1], row[2], row[3]) for row in found] == [
        ("июль 2026", "35", "36"),
        ("июль 2026", "35", "36"),
        ("май 2026", "40", "41"),
    ]
