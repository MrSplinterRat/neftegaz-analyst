"""Сторож поискового сервиса: адресат вопроса обязан быть один и назван.

★ЧТО ИМЕННО СТЕРЕЖЁТСЯ. Не «поиск работает» — это проверяется прогоном, а
сеть в тестах недоступна и не должна быть нужна. Стережётся ФОРМА НАСТРОЙКИ:
возврат к режиму `auto` или к списку из нескольких сервисов означает, что текст
вопроса пользователя снова уезжает туда, куда мы не выбирали. Такой откат не
ломает ни одного другого теста и не виден в интерфейсе — поэтому нужен сторож.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from neftegaz.tools import web
from neftegaz.tools.web import FORBIDDEN_BACKENDS, checked_backend, search_web_with_status


def _with_backend(monkeypatch, value: str) -> None:
    monkeypatch.setattr(web, "settings", replace(web.settings, web_backend=value))


def test_the_default_backend_is_one_named_service():
    """Умолчание — ровно одно имя, а не режим выбора и не список."""
    backend, why = checked_backend()
    assert why == ""
    assert backend not in FORBIDDEN_BACKENDS
    assert "," not in backend
    assert backend == "brave", "смена сервиса требует нового замера, а не правки теста"


def test_the_default_backend_exists_in_the_library():
    """★Имя сверяется со списком САМОЙ библиотеки, а не с нашей копией.

    Копия списка отстанет от библиотеки при обновлении и снова начнёт врать —
    ровно тем же способом, каким врал режим `auto`.
    """
    ddgs_engines = pytest.importorskip("ddgs.engines")
    backend, _ = checked_backend()
    assert backend in ddgs_engines.ENGINES["text"]


@pytest.mark.parametrize("value", ["auto", "all", "", "  "])
def test_a_backend_that_delegates_the_choice_is_refused(monkeypatch, value):
    """`auto` — это «библиотека решит», то есть семь адресатов вместо одного."""
    _with_backend(monkeypatch, value)
    backend, why = checked_backend()
    assert backend == ""
    assert "WEB_BACKEND" in why


def test_a_list_of_several_services_is_refused(monkeypatch):
    """Список тоже делает маршрут неизвестным до запроса."""
    _with_backend(monkeypatch, "brave,duckduckgo")
    backend, why = checked_backend()
    assert backend == ""
    assert "несколько" in why


def test_an_unknown_name_is_refused_instead_of_falling_back(monkeypatch):
    """★Главный случай: опечатка НЕ должна тихо включать режим auto.

    Библиотека на неизвестное имя возвращается к `auto` и печатает предупреждение
    в лог — то есть опечатка выглядит как работающий поиск. Здесь она обязана
    выглядеть как отказ.
    """
    pytest.importorskip("ddgs.engines")
    _with_backend(monkeypatch, "brve")
    backend, why = checked_backend()
    assert backend == ""
    assert "auto" in why, "отказ обязан объяснить, чем опасно оставить как есть"


def test_a_bad_setting_stops_the_search_instead_of_widening_it(monkeypatch):
    """Неверная настройка — отказ поиска, а не рассылка вопроса по семи сервисам.

    ★Проверка не делит механизм с проверяемым: сеть здесь не нужна вовсе —
    если бы запрос всё-таки ушёл, ответ не был бы `unavailable`.
    """
    _with_backend(monkeypatch, "auto")
    results, status = search_web_with_status("прогноз цены Brent")
    assert results == []
    assert status.startswith("unavailable:")
    assert "WEB_BACKEND" in status


def test_the_chosen_backend_reaches_the_library(monkeypatch):
    """Выбранное имя действительно едет в запрос, а не остаётся в настройке.

    ★Свидетель — аргумент, полученный библиотекой, а не наше намерение его
    передать. Без этой проверки настройка могла бы читаться, проверяться и
    молча не применяться.
    """
    seen: dict[str, object] = {}

    class _FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def text(self, query, **kwargs):
            seen.update(kwargs)
            seen["query"] = query
            return []

    import ddgs

    monkeypatch.setattr(ddgs, "DDGS", _FakeDDGS)
    _with_backend(monkeypatch, "mojeek")
    search_web_with_status("прогноз цены Brent")
    assert seen.get("backend") == "mojeek"
