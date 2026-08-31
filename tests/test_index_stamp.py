"""Отметка о правилах сборки индекса и сверка её с текущей настройкой."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from neftegaz.rag import index_stamp
from neftegaz.rag.index_stamp import STAMPED_FIELDS, mismatches, read_stamp, write_stamp


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Своё хранилище, чтобы тест не трогал боевую отметку.

    autouse, а не аргумент: фикстура нужна ради побочного действия — подмены
    пути, — и ни одному тесту её значение не требуется.
    """
    monkeypatch.setattr(
        index_stamp,
        "settings",
        replace(index_stamp.settings, qdrant_path=str(tmp_path / "qdrant")),
    )


def test_no_stamp_is_not_the_same_as_no_mismatch():
    """★Отсутствие отметки — это «сравнить не с чем», а не «всё сходится».

    Отличить одно от другого обязан вызывающий: `read_stamp() is None`.
    Если бы `mismatches()` было единственным ответом, ненайденная отметка
    читалась бы как согласие — то самое «я не смог», выданное за «в порядке».
    """
    assert read_stamp() is None
    assert mismatches() == []


def test_a_stamp_written_now_matches_the_current_settings():
    write_stamp(9450)
    assert mismatches() == []
    stamp = read_stamp()
    assert stamp["chunks"] == 9450
    assert set(STAMPED_FIELDS) <= set(stamp)


def test_a_changed_corpus_setting_shows_up_as_a_mismatch(monkeypatch):
    write_stamp(100)
    monkeypatch.setattr(
        index_stamp,
        "settings",
        replace(index_stamp.settings, chunk_size=999),
    )
    diff = mismatches()
    assert [name for name, _, _ in diff] == ["chunk_size"]
    assert diff[0][1] != 999 and diff[0][2] == 999


def test_every_stamped_field_is_actually_compared(monkeypatch):
    """★Список полей важнее любой его строки: забытое поле меняется молча."""
    write_stamp(1)
    for name in STAMPED_FIELDS:
        current = getattr(index_stamp.settings, name)
        changed = current + 1 if isinstance(current, int) else f"{current}-другое"
        monkeypatch.setattr(
            index_stamp, "settings", replace(index_stamp.settings, **{name: changed})
        )
        assert [x[0] for x in mismatches()] == [name], f"поле {name} не сверяется"
        monkeypatch.setattr(
            index_stamp, "settings", replace(index_stamp.settings, **{name: current})
        )


def test_a_broken_stamp_reads_as_absent_and_not_as_agreement():
    """Испорченный файл не должен читаться как «правила совпали»."""
    write_stamp(1)
    index_stamp.stamp_path().write_text("{это не json", encoding="utf-8")
    assert read_stamp() is None
    assert mismatches() == []


def test_the_stamp_lives_beside_the_store_not_inside_it():
    """Внутри каталога Qdrant посторонний файл — чужая территория."""
    path = write_stamp(1)
    assert path.name == "index-settings.json"
    assert "qdrant" not in path.parent.name
    assert json.loads(path.read_text(encoding="utf-8"))["chunks"] == 1
