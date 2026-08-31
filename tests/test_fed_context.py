"""Что именно считается «поданным в контекст» — проверка на самом тракте."""

from __future__ import annotations

from dataclasses import dataclass

from neftegaz.agent.graph import (
    _format_report_context,
    fed_report_hits,
)
from neftegaz.config import settings


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

    ⚠Длину задаёт не бюджет, а `settings.fragment_cap_chars`: каждый фрагмент сперва
    режется до 1800 знаков, и «половина бюджета» на входе превращается в 1800
    на выходе. Первая версия этого теста считала по бюджету и ошиблась именно
    здесь — три «огромных» фрагмента спокойно влезали втроём.
    """
    big = "я" * (settings.fragment_cap_chars * 2)
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


# ── подключение разговоров: слияние на общей шкале (Ш3) ───────────────────


from neftegaz.agent.graph import merge_borrowed  # noqa: E402


def _hit(chunk_id: str, score: float) -> FakeHit:
    return FakeHit(text=f"текст {chunk_id}", chunk_id=chunk_id, score=score)


def test_without_borrowed_fragments_nothing_changes():
    base = [_hit("a", 0.7), _hit("b", 0.6)]
    assert merge_borrowed(base, []) == base


def test_a_stronger_borrowed_fragment_enters_and_the_weakest_leaves():
    base = [_hit("a", 0.70), _hit("b", 0.61), _hit("c", 0.65)]
    merged = merge_borrowed(base, [_hit("x", 0.68)])
    assert [h.chunk_id for h in merged] == ["a", "x", "c"]
    assert len(merged) == len(base), "длина выдачи не растёт: кто вошёл, тот кого-то вытеснил"


def test_a_weaker_borrowed_fragment_does_not_enter():
    """★Никакого бонуса за подключение: не превзошёл — не вошёл."""
    base = [_hit("a", 0.70), _hit("b", 0.61), _hit("c", 0.65)]
    assert merge_borrowed(base, [_hit("x", 0.55)]) == base


def test_displacement_takes_the_weakest_by_score_not_the_last_in_order():
    """★Дефект, пойманный отрицательным контролем на боевом корпусе.

    Порядок обычных находок задан слиянием двух ветвей поиска, а не косинусом,
    поэтому последний в списке может быть сильнее середины. Первая версия
    роняла ХВОСТ — и фрагмент с оценкой 0.6592 вытеснил фрагмент с 0.6842,
    выиграв позицией вместо меры. Это и есть скрытый бонус.
    """
    base = [_hit("a", 0.67), _hit("weakest", 0.6485), _hit("c", 0.69), _hit("tail", 0.6842)]
    merged = merge_borrowed(base, [_hit("x", 0.6592)])
    ids = [h.chunk_id for h in merged]
    assert "tail" in ids, "сильнейший хвост обязан уцелеть"
    assert "weakest" not in ids, "вытесняется слабейший по оценке"
    assert "x" in ids


def test_a_borrowed_fragment_already_present_is_not_duplicated():
    base = [_hit("a", 0.70), _hit("b", 0.61)]
    merged = merge_borrowed(base, [_hit("a", 0.70)])
    assert [h.chunk_id for h in merged] == ["a", "b"]


def test_borrowed_fragments_fill_an_empty_result():
    """Когда обычный поиск не дал ничего, подключённые заполняют выдачу."""
    merged = merge_borrowed([], [_hit("x", 0.7), _hit("y", 0.6)])
    assert [h.chunk_id for h in merged] == ["x", "y"]
