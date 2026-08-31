"""Проверка живости обязана уметь сказать «нет».

★Что здесь проверяется и почему именно это. Прежняя проверка контейнера
смотрела на страницу Streamlit и объявляла здоровым контейнер, в котором не
работал ни один из двух узлов (26.08.2026 — кэш модели при файловой системе
только для чтения). Значит проверять надо не «возвращает ли она ноль», а
«различает ли она отказ»: половина тестов ниже — отказы, и каждый обязан
получить свой признак, а не общее «что-то не так».

Замер на трёх классах отказа: старая проверка различала 1 из 3, новая — 3 из 3,
и на исправной системе по-прежнему говорит «здоров».
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neftegaz import health


@pytest.fixture
def marker(tmp_path) -> Path:
    return tmp_path / ".readiness.json"


def _readiness(**over) -> health.Readiness:
    base = {
        "ok": True,
        "storage_ok": True,
        "embedder_ok": True,
        "chunks": 9450,
        "storage_error": "",
        "embedder_error": "",
        "checked_at": 1756600000.0,
    }
    base.update(over)
    base["ok"] = base["storage_ok"] and base["embedder_ok"]
    return health.Readiness(**base)


# ── отметка: круг записи и чтения ──────────────────────────────────────────


def test_marker_survives_a_round_trip(marker, monkeypatch):
    monkeypatch.setattr(health, "probe", lambda: _readiness())
    written = health.run_probe_and_record(marker)
    read_back = health.read_marker(marker)

    assert written.ok
    assert read_back == written


def test_a_missing_marker_is_not_health(marker):
    """★Отсутствие отметки — «ещё не готов», а не «здоров».

    Обратное решение и есть исходная ошибка: контейнер, о состоянии которого
    ничего не известно, объявлялся исправным.
    """
    assert health.read_marker(marker) is None


def test_a_corrupt_marker_is_not_health(marker):
    """Порча файла не должна превращаться в зелёный свет."""
    marker.write_text("{это не json", encoding="utf-8")
    assert health.read_marker(marker) is None


def test_a_marker_missing_a_field_is_not_health(marker):
    """Неполная отметка — тоже отказ: молчаливое умолчание врало бы о состоянии."""
    marker.write_text(json.dumps({"ok": True}), encoding="utf-8")
    assert health.read_marker(marker) is None


# ── каждый отказ назван своим именем ───────────────────────────────────────


def test_storage_failure_is_named(marker, monkeypatch):
    monkeypatch.setattr(
        health, "probe", lambda: _readiness(storage_ok=False, storage_error="нет каталога")
    )
    result = health.run_probe_and_record(marker)

    assert not result.ok
    assert "хранилище" in result.as_line()
    assert "нет каталога" in result.as_line()
    # ★Второй узел назван исправным, а не свален в общее «что-то сломалось»:
    # администратору чинить том с данными, а не кэш модели.
    assert "эмбеддер:" not in result.as_line()


def test_embedder_failure_is_named(marker, monkeypatch):
    monkeypatch.setattr(
        health,
        "probe",
        lambda: _readiness(embedder_ok=False, embedder_error="только для чтения"),
    )
    result = health.run_probe_and_record(marker)

    assert not result.ok
    assert "эмбеддер" in result.as_line()
    assert "только для чтения" in result.as_line()
    assert "хранилище:" not in result.as_line()


def test_a_healthy_probe_says_so_with_numbers(marker, monkeypatch):
    """★Отрицательный контроль ко всему файлу: исправная система даёт «здоров».

    Без него проверка, отвечающая «не здоров» ВСЕГДА, прошла бы все тесты выше
    и была бы не лучше отсутствующей.
    """
    monkeypatch.setattr(health, "probe", lambda: _readiness(chunks=9450))
    line = health.run_probe_and_record(marker).as_line()

    assert "готовность:" in line
    assert "9450" in line
    assert "НЕ достигнута" not in line


# ── проба трогает узлы по-настоящему ───────────────────────────────────────


def test_probe_reports_a_broken_store_instead_of_raising(monkeypatch):
    """Неготовность возвращается значением, а не исключением.

    Проверка, падающая с трассировкой, неотличима от сломанной проверки: и то,
    и другое даёт ненулевой код возврата и мусор в журнале.
    """

    class _Broken:
        def count(self):
            raise RuntimeError("каталог занят другим процессом")

        def _embed_query(self, text):  # noqa: ARG002 — подпись повторяет настоящую
            raise RuntimeError("модель не загрузилась")

    monkeypatch.setattr("neftegaz.rag.store.get_store", lambda: _Broken())
    result = health.probe()

    assert not result.ok
    assert not result.storage_ok and not result.embedder_ok
    assert "занят другим процессом" in result.storage_error
    assert "не загрузилась" in result.embedder_error


def test_probe_calls_the_real_embedding_path(monkeypatch):
    """★Проба ЗОВЁТ тракт, а не повторяет его своим вызовом fastembed.

    Копия тракта отстала бы молча — она не подаёт признаков устаревания. Здесь
    это закреплено: подменённый метод хранилища обязан быть вызван, и именно с
    вычислением вектора, а не с созданием объекта модели.
    """
    called = []

    class _Store:
        def count(self):
            return 7

        def _embed_query(self, text):
            called.append(text)
            return [0.1, 0.2]

    monkeypatch.setattr("neftegaz.rag.store.get_store", lambda: _Store())
    result = health.probe()

    assert result.ok
    assert result.chunks == 7
    assert called == [health.PROBE_TEXT]


def test_an_empty_vector_is_a_failure(monkeypatch):
    """Пустой вектор — отказ, а не «ноль признаков»."""

    class _Store:
        def count(self):
            return 7

        def _embed_query(self, text):  # noqa: ARG002 — подпись повторяет настоящую
            return []

    monkeypatch.setattr("neftegaz.rag.store.get_store", lambda: _Store())
    result = health.probe()

    assert not result.ok
    assert not result.embedder_ok
    assert result.storage_ok


# ── скрипт проверки живости ────────────────────────────────────────────────


def _healthcheck_module():
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "scripts" / "healthcheck.py"
    spec = importlib.util.spec_from_file_location("healthcheck_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_healthcheck_fails_when_the_page_is_down(monkeypatch, capsys):
    module = _healthcheck_module()
    monkeypatch.setattr(module, "page_is_up", lambda url=None: "страница не отвечает: отказ")

    assert module.main() == 1
    assert "страница не отвечает" in capsys.readouterr().err


def test_healthcheck_fails_when_the_page_is_up_but_nodes_are_not(monkeypatch, capsys):
    """★Ровно тот случай 26.08: страница отвечает, система не работает."""
    module = _healthcheck_module()
    monkeypatch.setattr(module, "page_is_up", lambda url=None: "")
    monkeypatch.setattr(
        module, "read_marker", lambda: _readiness(embedder_ok=False, embedder_error="нет прав")
    )

    assert module.main() == 1
    assert "эмбеддер" in capsys.readouterr().err


def test_healthcheck_fails_while_the_probe_has_not_finished(monkeypatch, capsys):
    module = _healthcheck_module()
    monkeypatch.setattr(module, "page_is_up", lambda url=None: "")
    monkeypatch.setattr(module, "read_marker", lambda: None)

    assert module.main() == 1
    assert "ещё не завершилась" in capsys.readouterr().err


def test_healthcheck_passes_on_a_working_system(monkeypatch, capsys):
    """★Отрицательный контроль: проверка, всегда красная, не лучше отсутствующей."""
    module = _healthcheck_module()
    monkeypatch.setattr(module, "page_is_up", lambda url=None: "")
    monkeypatch.setattr(module, "read_marker", lambda: _readiness(chunks=9450))

    assert module.main() == 0
    assert "9450" in capsys.readouterr().out
