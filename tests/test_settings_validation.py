"""Проверка настроек при старте (Р-014).

★ЗАЧЕМ. Опечатка в `.env` проявлялась не отказом, а тихой сменой поведения:
`ELASTICITY_SOURCE=mesured` переключал расчёт с эластичности, измеренной на
корпусе, на литературную — то есть менял ЧИСЛА в ответе и не говорил ни слова.
Второй случай того же рода (`CONVERSATION_MEMORY`) падал, но при первом
использовании: система поднималась, показывала интерфейс и умирала на первом
вопросе.

★ЧЕГО ЗДЕСЬ СТЕРЕГУТСЯ ДВЕ ЛОВУШКИ, ОБЕ УЖЕ ЛОВЛЕННЫЕ.
1. Проверка, ПРОХОДЯЩАЯ всегда, не лучше отсутствующей. Поэтому на каждое поле
   с границами есть диверсия: кривое значение обязано быть поймано, а сообщение
   обязано называть имя переменной окружения.
2. Проверка, ПАДАЮЩАЯ всегда, хуже отсутствующей. Поэтому первым идёт тест, что
   настоящая настройка проекта поднимается без единой находки.

★И третья, особая: реестр проверок обязан РАСТИ вместе с `Settings`. Поле,
добавленное без записи в реестре, выпало бы из проверки молча — и мы бы об этом
узнали ровно тогда, когда в нём завелась бы опечатка.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

from neftegaz import config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKED = [spec for spec in config.SETTING_SPECS if spec.kind != "free"]
FREE = [spec for spec in config.SETTING_SPECS if spec.kind == "free"]


def junk_for(spec) -> object:
    """Заведомо непригодное значение для этой настройки.

    Выводится из границ самой записи, а не из моего представления о ней: список
    руками разошёлся бы с реестром при первой же новой настройке.
    """
    if spec.kind in ("int", "float"):
        if spec.low is not None:
            return spec.low - 1
        return (spec.high or 0) + 1
    if spec.kind == "choice":
        return "такого-значения-нет"
    if spec.kind == "bool":
        return "может быть"
    return "///не по форме///"


# ── проверка не падает на верной настройке ─────────────────────────────────


def test_the_real_settings_of_this_project_pass():
    """★Отрицательный контроль к самой проверке: она обязана уметь молчать."""
    assert config.check_settings(config.settings) == []


def test_the_example_env_shipped_with_the_project_is_valid():
    """`.env.example` — то, что человек копирует первым движением.

    Если бы наша же проверка его не принимала, установка по инструкции падала бы
    на первом шаге.
    """
    example = PROJECT_ROOT / ".env.example"
    assert example.is_file(), ".env.example отсутствует — копировать нечего"
    environment = dict(os.environ)
    for line in example.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        environment[name.strip()] = value.strip().strip('"').strip("'")
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import neftegaz.config as c; print('ПОДНЯЛОСЬ', len(c.SETTING_SPECS))",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=environment,
        timeout=120,
    )
    assert "ПОДНЯЛОСЬ" in done.stdout, f"настройка из .env.example отвергнута:\n{done.stderr}"


# ── проверка умеет упасть, и по каждому полю ───────────────────────────────


@pytest.mark.parametrize("spec", CHECKED, ids=[spec.env for spec in CHECKED])
def test_every_checked_field_rejects_an_unusable_value(spec):
    """Диверсия по каждому полю с границами — и имя переменной в сообщении.

    Человек правит `.env`, где нет ни `min_score`, ни `conversation_memory`.
    Сообщение, которое нельзя перенести в правку одним движением, — это
    полсообщения.
    """
    broken = dataclasses.replace(config.settings, **{spec.field: junk_for(spec)})
    problems = config.check_settings(broken)
    assert problems, f"{spec.env}: непригодное значение прошло молча"
    assert any(spec.env in problem for problem in problems), (
        f"находка есть, но имени переменной {spec.env} в ней нет: {problems}"
    )


@pytest.mark.parametrize("spec", FREE, ids=[spec.env for spec in FREE])
def test_every_free_field_says_why_there_is_nothing_to_check(spec):
    """★«Проверять нечего» обязано быть написано вслух.

    Иначе «у этого поля нет замкнутого множества значений» и «проверить забыли»
    выглядят в коде одинаково.
    """
    assert spec.note, f"{spec.env}: поле объявлено свободным без причины"


# ── реестр обязан расти вместе с настройками ───────────────────────────────


def test_the_registry_covers_every_field_of_settings():
    fields = {field.name for field in dataclasses.fields(config.Settings)}
    covered = {spec.field for spec in config.SETTING_SPECS}
    assert fields - covered == set(), "поля без записи в реестре проверок"
    assert covered - fields == set(), "записи о полях, которых в Settings нет"


def test_editable_parameters_are_a_subset_of_the_registry():
    """★Границы интерфейса и границы старта — ОДНИ. Две копии разошлись бы молча."""
    assert set(config.TURN_PARAMETERS) <= set(config.SETTING_SPECS)
    assert all(spec.editable for spec in config.TURN_PARAMETERS)


def test_the_check_is_not_trivially_empty():
    """Содержательность закреплена числом: проверяемых полей заметно больше нуля."""
    assert len(CHECKED) >= 20, f"проверяемых полей осталось {len(CHECKED)} — проверка выродилась"


# ── связи между полями ─────────────────────────────────────────────────────


def test_overlap_not_smaller_than_the_chunk_is_caught():
    """Каждое значение по отдельности пригодно, а пара — нет."""
    broken = dataclasses.replace(config.settings, chunk_size=1000, chunk_overlap=1000)
    problems = config.check_settings(broken)
    assert any("CHUNK_OVERLAP" in problem for problem in problems), problems


def test_long_horizon_shorter_than_the_short_one_is_caught():
    broken = dataclasses.replace(config.settings, elasticity_short_days=90, elasticity_long_days=30)
    problems = config.check_settings(broken)
    assert any("ELASTICITY_LONG_DAYS" in problem for problem in problems), problems


def test_all_problems_are_reported_at_once():
    """★Все находки разом: `.env` чинится за один заход, а не за пять запусков."""
    broken = dataclasses.replace(config.settings, top_k=-5, min_score=9.0, web_region="Россия")
    problems = config.check_settings(broken)
    assert len(problems) >= 3, problems


# ── и настоящий старт, а не только вызов функции ───────────────────────────


def test_a_typo_in_elasticity_source_stops_the_start():
    """★Тот самый случай, ради которого проверка заведена.

    Проверяется НАСТОЯЩИЙ старт в отдельном процессе, а не вызов функции: если
    бы проверку забыли позвать при импорте, вызов функции всё равно проходил бы,
    и тест был бы зелёным ровно там, где система тихо работает не так.
    """
    environment = {**os.environ, "ELASTICITY_SOURCE": "mesured"}
    done = subprocess.run(
        [sys.executable, "-c", "import neftegaz.config"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=environment,
        timeout=120,
    )
    assert done.returncode != 0, "опечатка в ELASTICITY_SOURCE не остановила старт"
    assert "ELASTICITY_SOURCE" in done.stderr
    assert "measured" in done.stderr, "сообщение не называет допустимые значения"


def test_a_correct_value_still_starts():
    """Обратная сторона: верное значение обязано подниматься."""
    environment = {**os.environ, "ELASTICITY_SOURCE": "literature"}
    done = subprocess.run(
        [sys.executable, "-c", "import neftegaz.config; print('ПОДНЯЛОСЬ')"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=environment,
        timeout=120,
    )
    assert "ПОДНЯЛОСЬ" in done.stdout, done.stderr


# ── бюджеты контекста против окна модели: предупреждение, а не отказ ───────


def test_budgets_too_large_for_the_declared_window_are_warned_about():
    """★Бюджеты заданы в ЗНАКАХ, окно модели меряется в ТОКЕНАХ.

    Отношение между ними зависит от языка: замер на нашем материале дал 2.48
    знака на токен по английским таблицам STEO и 2.02 по русским ответам. Сумма
    бюджетов в 15 000 знаков — это 6000–7400 токенов только контекста, и при
    окне 8k сервер модели отвечает отказом посреди работы.
    """
    tight = dataclasses.replace(config.settings, llm_context_tokens=8192)
    warning = config.context_budget_warning(tight)
    assert warning is not None, "тесное окно не вызвало предупреждения"
    assert "8192" in warning
    assert "REPORT_BUDGET_CHARS" in warning, "предупреждение не говорит, что именно править"


def test_a_roomy_window_produces_no_warning():
    """★Отрицательный контроль: предупреждение, звучащее всегда, не слышат."""
    roomy = dataclasses.replace(config.settings, llm_context_tokens=32768)
    assert config.context_budget_warning(roomy) is None


def test_an_unknown_window_is_not_guessed():
    """0 значит «окно неизвестно». Выдумать его за пользователя хуже, чем молчать."""
    unknown = dataclasses.replace(config.settings, llm_context_tokens=0)
    assert config.context_budget_warning(unknown) is None


def test_the_warning_does_not_stop_the_start():
    """★Это ПРЕДУПРЕЖДЕНИЕ, а не отказ, и разница принципиальная.

    Перевод знаков в токены оценочный: считает его токенизатор чужого семейства,
    а модель настраивается и может быть любой. Уронить старт на такой оценке
    значило бы поменять один тихий отказ на другой, громкий и часто ложный.
    """
    environment = {**os.environ, "LLM_CONTEXT_TOKENS": "8192"}
    done = subprocess.run(
        [sys.executable, "-c", "import neftegaz.config; print('ПОДНЯЛОСЬ')"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=environment,
        timeout=120,
    )
    assert "ПОДНЯЛОСЬ" in done.stdout, done.stderr
    assert "превышена длина контекста" in done.stdout, "предупреждение не напечатано при старте"
