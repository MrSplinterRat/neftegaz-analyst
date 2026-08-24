"""Tests for chunking, citation formatting, routing and the web filter.

The citation tests are the load-bearing ones. A wrong number in a forecast is a
bad forecast; a wrong or invented citation makes an unverified claim look
verified, which is the failure this product exists to prevent.
"""

from __future__ import annotations

import pytest

from neftegaz.agent.graph import parse_horizon_days, parse_supply_change
from neftegaz.rag.chunking import chunk_document, chunk_pages
from neftegaz.rag.ingest import parse_filename
from neftegaz.tools.citations import format_answer, format_claim
from neftegaz.tools.web import DENY_DOMAINS, WebResult, _domain, _is_denied, _is_preferred


# ── chunking ───────────────────────────────────────────────────────────────


def test_chunks_cover_every_character():
    pages = [{"page": 1, "text": "a" * 100}, {"page": 2, "text": "b" * 100}]
    chunks = chunk_pages(pages, size=60, overlap=10)
    rebuilt = chunks[0]["text"]
    for chunk in chunks[1:]:
        rebuilt += chunk["text"][10:]  # drop the overlap
    assert rebuilt == "a" * 100 + "b" * 100


def test_chunk_spanning_a_page_break_reports_both_pages():
    pages = [{"page": 1, "text": "a" * 50}, {"page": 2, "text": "b" * 50}]
    chunks = chunk_pages(pages, size=80, overlap=0)
    assert chunks[0]["page_start"] == 1
    assert chunks[0]["page_end"] == 2


def test_empty_page_in_the_middle_does_not_shift_numbering():
    """A chart-only page yields no text but must not renumber what follows."""
    pages = [
        {"page": 5, "text": "x" * 30},
        {"page": 6, "text": ""},
        {"page": 8, "text": "y" * 30},
    ]
    chunks = chunk_pages(pages, size=30, overlap=0)
    assert chunks[0]["page_start"] == 5
    assert chunks[1]["page_start"] == 8


def test_overlap_must_be_smaller_than_size():
    """Otherwise the window never advances and the loop never terminates."""
    with pytest.raises(ValueError):
        chunk_pages([{"page": 1, "text": "abc"}], size=10, overlap=10)


def test_chunk_document_stamps_citation_metadata():
    chunks = chunk_document(
        [{"page": 3, "text": "z" * 40}], source_name="EIA STEO", doc_date="июль 2026", size=40, overlap=0
    )
    assert chunks[0]["source_name"] == "EIA STEO"
    assert chunks[0]["date"] == "июль 2026"
    assert chunks[0]["page"] == 3


# ── filename metadata ──────────────────────────────────────────────────────


def test_filename_with_year_and_month():
    meta = parse_filename("/data/reports/EIA_STEO_2026-07.pdf")
    assert meta.source_name == "EIA STEO"
    assert meta.date == "июль 2026"


def test_filename_with_year_only():
    meta = parse_filename("OPEC_Annual_2025.pdf")
    assert meta.source_name == "OPEC Annual"
    assert meta.date == "2025"


def test_filename_without_date_still_yields_a_name():
    """Indexed and cited without a date is honest; refusing to index is not."""
    meta = parse_filename("some_report.pdf")
    assert meta.source_name == "some report"
    assert meta.date == ""


# ── citations ──────────────────────────────────────────────────────────────


def test_report_citation_exact_format():
    claim = {
        "source_type": "report",
        "text": "Спрос вырастет на 1.2 млн барр./сут.",
        "source_name": "OPEC MOMR",
        "date": "март 2025",
        "page": 14,
    }
    assert format_claim(claim) == (
        "Спрос вырастет на 1.2 млн барр./сут. [Отчёт OPEC MOMR, март 2025, с. 14]"
    )


def test_web_citation_exact_format():
    claim = {"source_type": "web", "text": "Brent торгуется у 92 долл.", "source_name": "Reuters"}
    assert format_claim(claim) == "Brent торгуется у 92 долл. [Источник: Reuters, web]"


def test_unknown_source_type_is_refused():
    with pytest.raises(ValueError):
        format_claim({"source_type": "guess", "text": "x", "source_name": "y"})


def test_report_claim_without_page_is_refused():
    """A page cannot be invented: it is what makes the citation checkable."""
    with pytest.raises(KeyError):
        format_claim(
            {"source_type": "report", "text": "x", "source_name": "OPEC", "date": "март 2025"}
        )


def test_mixed_answer_marks_each_paragraph_separately():
    answer = format_answer(
        [
            {
                "source_type": "report",
                "text": "Запасы снизились.",
                "source_name": "EIA STEO",
                "date": "июль 2026",
                "page": 7,
            },
            {"source_type": "web", "text": "Цена выросла сегодня.", "source_name": "Reuters"},
        ]
    )
    assert "[Отчёт EIA STEO, июль 2026, с. 7]" in answer
    assert "[Источник: Reuters, web]" in answer
    assert answer.count("\n\n") == 1


# ── web filtering ──────────────────────────────────────────────────────────


def test_domain_extraction_strips_www():
    assert _domain("https://www.reuters.com/business/energy") == "reuters.com"


def test_tabloid_is_denied_including_subdomains():
    assert _is_denied("dailymail.co.uk")
    assert _is_denied("news.dailymail.co.uk")


def test_unknown_domain_is_allowed_but_not_preferred():
    assert not _is_denied("some-energy-blog.example")
    assert not _is_preferred("some-energy-blog.example")


def test_agency_domain_is_preferred():
    assert _is_preferred("eia.gov")
    assert _is_preferred("www.reuters.com".removeprefix("www."))


def test_deny_list_and_preferred_list_do_not_overlap():
    """A domain in both lists would make ranking depend on evaluation order."""
    from neftegaz.tools.web import PREFERRED_DOMAINS

    assert not (DENY_DOMAINS & PREFERRED_DOMAINS)


def test_web_result_becomes_a_web_claim():
    result = WebResult(
        title="Reuters", url="https://reuters.com/x", snippet="текст", domain="reuters.com", preferred=True
    )
    claim = result.as_claim()
    assert claim["source_type"] == "web"
    assert format_claim(claim).endswith("[Источник: Reuters, web]")


# ── request parsing ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "question,expected",
    [
        ("спрогнозируй Brent на 3 месяца", 90),
        ("прогноз на 2 недели", 14),
        ("что будет через 1 год", 365),
        ("оцени на 45 дней", 45),
        ("спрогнозируй цену", 90),  # default
        ("прогноз на 5 лет", 730),  # clamped
    ],
)
def test_horizon_parsing(question, expected):
    assert parse_horizon_days(question) == expected


@pytest.mark.parametrize(
    "question,expected",
    [
        ("при сокращении добычи на 1.5 млн барр/сут", -1.5),
        ("если добыча вырастет на 2 млн баррелей в сутки", 2.0),
        ("что с ценой Brent", 0.0),
    ],
)
def test_supply_scenario_parsing(question, expected):
    assert parse_supply_change(question) == pytest.approx(expected)
