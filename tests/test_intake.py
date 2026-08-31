"""Tests for intake acceptance.

The point of the module is that it always answers, so the tests are mostly
about *damage*: a truncated file, a header that is not there, an offset that
points past the end. Each case is built as bytes rather than taken from the
corpus — real files cannot be broken on demand, and a check that has never
seen a broken file is a check nobody has tested.

The corpus test at the end is skipped when data/reports/ is absent, so the
suite runs on a clean checkout.
"""

from __future__ import annotations

import glob
import os

import pytest

from neftegaz.rag.intake import (
    BROKEN,
    NOTICE,
    OK,
    inspect_directory,
    inspect_pdf,
)

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "reports")

# Минимальный PDF, который poppler читает: один пустой лист. Держим его в
# тесте байтами, чтобы порча была ТОЧЕЧНОЙ — меняется ровно то, что проверяем.
# ★Пробел в конце строк xref записан как \x20 НАМЕРЕННО. По спецификации PDF
# запись таблицы xref занимает РОВНО 20 байт, и последний из них — этот пробел:
# он несущий, а не оформительский. Невидимый пробел в конце строки любой
# форматтер или линтер однажды уберёт «за чистоту», и фикстура перестанет быть
# валидным PDF — тихо, потому что все тесты продолжат запускаться.
MINIMAL_PDF = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj
xref
0 4
0000000000 65535 f\x20
0000000009 00000 n\x20
0000000056 00000 n\x20
0000000111 00000 n\x20
trailer<</Size 4/Root 1 0 R>>
startxref
183
%%EOF
"""


def write(tmp_path, name: str, data: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


# ── контейнер ──────────────────────────────────────────────────────────────


def test_missing_file_is_broken_not_an_exception():
    """The module's whole contract: it answers instead of raising."""
    report = inspect_pdf("/nonexistent/nope.pdf")
    assert report.severity == BROKEN
    assert not report.readable
    assert [f.code for f in report.findings] == ["unreadable"]


def test_empty_file(tmp_path):
    report = inspect_pdf(write(tmp_path, "empty.pdf", b""))
    assert report.severity == BROKEN
    assert "empty" in {f.code for f in report.findings}


def test_truncated_file_loses_eof_and_startxref(tmp_path):
    """The commonest damage in the wild — a download cut short."""
    report = inspect_pdf(write(tmp_path, "cut.pdf", MINIMAL_PDF[: len(MINIMAL_PDF) // 2]))
    codes = {f.code for f in report.findings}
    assert "no_eof" in codes
    assert "no_startxref" in codes
    assert report.severity == BROKEN


def test_header_must_be_near_the_start(tmp_path):
    """A PDF that begins with two kilobytes of something else is not a PDF."""
    data = b"\x00" * 2048 + MINIMAL_PDF
    report = inspect_pdf(write(tmp_path, "shifted.pdf", data))
    assert "no_header" in {f.code for f in report.findings}


def test_startxref_past_end_of_file(tmp_path):
    data = MINIMAL_PDF.replace(b"startxref\n183", b"startxref\n999999")
    report = inspect_pdf(write(tmp_path, "far.pdf", data))
    codes = {f.code for f in report.findings}
    assert "startxref_out_of_range" in codes
    assert report.severity == BROKEN


def test_startxref_inside_the_file_but_pointing_at_nothing(tmp_path):
    """Offset is plausible, target is not — the trace of a careless rewriter."""
    data = MINIMAL_PDF.replace(b"startxref\n183", b"startxref\n42")
    report = inspect_pdf(write(tmp_path, "misaligned.pdf", data))
    assert "startxref_misaligned" in {f.code for f in report.findings}


def test_healthy_minimal_pdf_reports_facts(tmp_path):
    report = inspect_pdf(write(tmp_path, "ok.pdf", MINIMAL_PDF))
    codes = {f.code for f in report.findings}
    assert "no_eof" not in codes
    assert "no_startxref" not in codes
    assert "startxref_out_of_range" not in codes
    assert report.facts["version"] == "1.4"
    assert report.facts["revisions"] == 1
    assert report.facts["size"] == len(MINIMAL_PDF)


def test_incremental_updates_are_a_notice_not_a_defect(tmp_path):
    """Legal, but it means the last revision has the final say about an object."""
    data = MINIMAL_PDF + b"\n" + MINIMAL_PDF
    report = inspect_pdf(write(tmp_path, "twice.pdf", data))
    notices = {f.code for f in report.findings if f.severity == NOTICE}
    assert "incremental_updates" in notices
    assert report.facts["revisions"] == 2


def test_encryption_is_flagged(tmp_path):
    data = MINIMAL_PDF.replace(b"trailer<</Size 4", b"trailer<</Encrypt 9 0 R/Size 4")
    report = inspect_pdf(write(tmp_path, "enc.pdf", data))
    assert "encrypted" in {f.code for f in report.findings}


# ── ступени и порядок ──────────────────────────────────────────────────────


def test_severity_is_the_worst_step_not_the_last_one(tmp_path):
    """A NOTICE after a BROKEN must not talk the report back down to readable."""
    data = MINIMAL_PDF[: len(MINIMAL_PDF) // 2]  # broken
    report = inspect_pdf(write(tmp_path, "mix.pdf", data))
    report.add("cosmetic", NOTICE, "приписано после находки-поломки")
    assert report.severity == BROKEN
    assert not report.readable


def test_report_serialises_to_a_dict():
    report = inspect_pdf("/nonexistent/nope.pdf")
    payload = report.to_dict()
    assert payload["severity"] == BROKEN
    assert payload["readable"] is False
    assert payload["findings"][0]["code"] == "unreadable"


def test_directory_scan_is_sorted_and_survives_a_missing_directory(tmp_path):
    write(tmp_path, "b.pdf", MINIMAL_PDF)
    write(tmp_path, "a.pdf", MINIMAL_PDF)
    write(tmp_path, "notes.txt", b"not a pdf")
    reports = inspect_directory(str(tmp_path))
    assert [os.path.basename(r.path) for r in reports] == ["a.pdf", "b.pdf"]
    assert inspect_directory(str(tmp_path / "absent")) == []


def test_two_runs_over_the_same_bytes_give_the_same_report(tmp_path):
    """Determinism is the property we actually promise, so it gets a test."""
    path = write(tmp_path, "same.pdf", MINIMAL_PDF)
    assert inspect_pdf(path).to_dict() == inspect_pdf(path).to_dict()


# ── корпус ─────────────────────────────────────────────────────────────────


# ★Условие пропуска смотрит на PDF, а не на каталог, и это исправление, а не
# придирка. Каталог `data/reports` существует ВСЕГДА: в git лежит `.gitkeep`,
# а сам корпус (сотни мегабайт отчётов) не трекается и приезжает отдельно.
# Условие «каталог есть» поэтому истинно и там, где корпуса нет, — и тест падал
# с «в data/reports нет PDF» в любой свежей копии дерева. То есть ОТСУТСТВИЕ
# ДАННЫХ выдавало себя за ПОРЧУ КОРПУСА: ровно тот класс, который этот файл и
# стережёт, только повёрнутый на саму приёмку. Проверено прогоном на отдельной
# рабочей копии той же вершины: 565 тестов зелены, красен один этот.
_SHIPPED_PDFS = sorted(glob.glob(os.path.join(REPORTS_DIR, "*.pdf")))


@pytest.mark.skipif(
    not _SHIPPED_PDFS, reason="корпус не выложен: в data/reports нет ни одного PDF"
)
def test_shipped_corpus_is_readable_and_its_caveats_are_named():
    """Our own corpus passes — but not silently: the font caveat must be stated.

    ★ЭТОТ ТЕСТ СТЕРЕЖЁТ ГЛАВНЫЙ ВЫВОД: «битых файлов нет» ≠ «читается верно».
    Файлы EIA STEO структурно чисты, и при этом почти все их шрифты идут без
    ToUnicode. Если однажды приёмка перестанет об этом говорить, отчёт станет
    выглядеть чище, чем корпус.
    """
    reports = inspect_directory(REPORTS_DIR)
    # Пустая выдача здесь уже НЕ означает «корпуса нет» — до этой строки мы
    # дошли только потому, что PDF в каталоге найдены. Значит их не вернул
    # обходчик, и это его отказ.
    assert reports, f"обходчик не вернул ни одного из {len(_SHIPPED_PDFS)} PDF в {REPORTS_DIR}"
    for report in reports:
        assert report.readable, report.summary()
        assert report.severity in (OK, NOTICE)
        assert "pages_extractor" in report.facts
        assert report.facts["pages_extractor"] > 0
        # Расхождение путей чтения — то, ради чего приёмка и заводится.
        assert "page_count_mismatch" not in {f.code for f in report.findings}
    assert any("fonts_without_tounicode" in {f.code for f in r.findings} for r in reports), (
        "оговорка про шрифты пропала из отчёта"
    )


# ── «пусто» против «не прочли» на уровне страницы ──────────────────────────


def test_blank_page_is_marked_unreadable_when_another_reader_sees_text(monkeypatch):
    """★Пустая страница у нас плюс текст у второго читателя = мы её потеряли.

    Изнутри одного разборщика эти случаи неразличимы: и пустая страница, и
    провал декодирования выглядят как отсутствие слов. Свидетель нужен снаружи.
    """
    from neftegaz.rag import ingest

    pages = [
        {"page": 1, "text": "есть текст"},
        {"page": 2, "text": "   "},
        {"page": 3, "text": ""},
    ]
    monkeypatch.setattr(
        "neftegaz.rag.crosscheck._read_poppler",
        lambda _path: ["есть текст", "здесь ЕСТЬ данные", ""],
    )
    marked = ingest.mark_unreadable_pages("нет-такого.pdf", pages)

    assert marked == 1
    assert pages[1].get("unreadable") is True, "потерянная страница не помечена"
    assert pages[1]["unreadable_witness"] == "poppler", "не назван свидетель"
    assert "unreadable" not in pages[2], "по-настоящему пустая страница помечена зря"
    assert "unreadable" not in pages[0], "непустая страница помечена зря"


def test_no_witness_means_no_verdict(monkeypatch):
    """★Свидетель не явился — пометки НЕТ, а не «страница пуста».

    Иначе отсутствие проверки читалось бы как её успешное прохождение: ровно
    та подмена, ради которой в системе заведена отдельная ступень «неизвестно».
    """
    from neftegaz.rag import ingest

    pages = [{"page": 1, "text": ""}]
    monkeypatch.setattr("neftegaz.rag.crosscheck._read_poppler", lambda _path: None)

    assert ingest.mark_unreadable_pages("нет-такого.pdf", pages) == 0
    assert "unreadable" not in pages[0], "без свидетеля вынесен вердикт"
