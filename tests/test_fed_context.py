"""Что именно считается «поданным в контекст» — проверка на самом тракте."""

from __future__ import annotations

from dataclasses import dataclass

from neftegaz.agent.graph import (
    FRAGMENT_CAP_CHARS,
    _format_report_context,
    fed_report_hits,
)


@dataclass(frozen=True)
class FakeHit:
    text: str
    chunk_id: str
    score: float = 0.9
    source_name: str = "EIA STEO"
    date: str = "июль 2026"
    page: int = 52
    page_end: int = 52
    context: str = ""


def test_every_hit_that_fits_is_reported_as_fed():
    hits = [FakeHit("текст " * 20, f"id{i}") for i in range(3)]
    assert [h.chunk_id for h in fed_report_hits(hits)] == ["id0", "id1", "id2"]


def _overflowing_hits(count: int = 5) -> list[FakeHit]:
    """Фрагменты, которых заведомо больше, чем влезает в бюджет.

    ⚠Длину задаёт не бюджет, а `FRAGMENT_CAP_CHARS`: каждый фрагмент сперва
    режется до 1800 знаков, и «половина бюджета» на входе превращается в 1800
    на выходе. Первая версия этого теста считала по бюджету и ошиблась именно
    здесь — три «огромных» фрагмента спокойно влезали втроём.
    """
    big = "я" * (FRAGMENT_CAP_CHARS * 2)
    return [FakeHit(big, f"id{i}") for i in range(count)]


def test_a_hit_that_did_not_fit_the_budget_is_not_reported_as_fed():
    """★Нашлось ≠ подано. Хвост, отброшенный бюджетом, модель не видела."""
    fed = [h.chunk_id for h in fed_report_hits(_overflowing_hits())]
    assert fed == ["id0", "id1", "id2"], f"в бюджет влезают три, а получили {fed}"


def test_the_trail_matches_what_the_context_actually_contains():
    """Прибор ЗОВЁТ тракт, а не повторяет его: списки обязаны сойтись.

    ★И проверка обязана быть содержательной: если бюджет ничего не отбросил,
    равенство выполнилось бы само собой и о согласии не сказало бы ничего.
    """
    hits = _overflowing_hits()
    rendered = _format_report_context(hits)
    fed = fed_report_hits(hits)
    assert rendered.count("[фрагмент:") == len(fed)
    assert len(fed) < len(hits), "бюджет обязан был отбросить хвост, иначе проверка пустая"
