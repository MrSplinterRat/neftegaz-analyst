"""Роль принадлежит заказчику, правила при отсутствии данных — нет.

★Что здесь проверяется. Заказчик вправе задать свой текст роли и свою реплику
отказа: тон общения и границы компетенции — его редакционная политика. Но на
правилах «не выдумывай числа» и «приоритет источников» стои́т сверка цитат и
весь раздел отчёта о проверяемости, поэтому они не заменяются ничем.

★Защита сделана устройством, а не проверкой текста, и главный тест здесь —
`test_mandatory_rules_survive_a_role_file_without_them`: файл заказчика может
не содержать обязательных правил вовсе, а в итоговом промпте они всё равно
будут, потому что приклеиваются кодом.

Замер: пять видов негодного файла отвергаются при старте с названной причиной,
законный принимается, текст доходит до промпта, обязательные правила уцелевают.
"""

from __future__ import annotations

import pytest

from neftegaz import config
from neftegaz.agent import prompts

# Куски, по которым узнаются обязательные правила. Не весь текст: тест не должен
# падать от правки формулировки, он про НАЛИЧИЕ правила, а не про его редакцию.
MANDATORY_MARKS = (
    "Никогда не выдумывай числа",
    "ПРИОРИТЕТ ИСТОЧНИКОВ",
    "Прогноз — это не факт",
)

ROLE_SPEC = next(spec for spec in config.SETTING_SPECS if spec.field == "system_prompt_file")
REPLY_SPEC = next(spec for spec in config.SETTING_SPECS if spec.field == "out_of_scope_file")


@pytest.fixture
def setting():
    """Подменить поле настроек и вернуть прежнее значение после теста.

    ★`Settings` — замороженная запись, и `monkeypatch.setattr` на ней падает.
    Обходим через `object.__setattr__` и возвращаем сами: тест, оставивший
    после себя изменённые настройки, портит соседние — а порядок тестов не
    гарантирован, и находка выглядела бы плавающей.
    """
    saved: dict[str, object] = {}

    def _set(name: str, value: object) -> None:
        saved.setdefault(name, getattr(config.settings, name))
        object.__setattr__(config.settings, name, value)

    yield _set

    for name, value in saved.items():
        object.__setattr__(config.settings, name, value)


@pytest.fixture
def role_file(tmp_path, setting):
    """Подставить файл роли и вернуть путь; настройка возвращается сама."""

    def _use(text: str) -> str:
        path = tmp_path / "роль.txt"
        path.write_text(text, encoding="utf-8")
        setting("system_prompt_file", str(path))
        return str(path)

    return _use


# ── негодный файл отвергается при старте, а не молча заменяется умолчанием ──


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(ValueError, match="нет или это не файл"):
        ROLE_SPEC.parse(str(tmp_path / "нет-такого.txt"))


def test_a_directory_is_refused(tmp_path):
    with pytest.raises(ValueError, match="нет или это не файл"):
        ROLE_SPEC.parse(str(tmp_path))


@pytest.mark.parametrize("content", ["", "   \n\n  "])
def test_an_empty_file_is_refused(tmp_path, content):
    """★Пустой файл — отказ, а не «текста нет».

    Молчаливый переход к умолчанию отдал бы заказчику нашу роль вместо его
    собственной, и он не узнал бы об этом никогда.
    """
    path = tmp_path / "пусто.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="пуст"):
        ROLE_SPEC.parse(str(path))


def test_an_oversized_file_is_refused(tmp_path):
    """Текст роли едет в КАЖДЫЙ запрос: его длина — вопрос бюджета контекста."""
    path = tmp_path / "огромный.txt"
    path.write_text("я" * (config.MAX_PROMPT_TEXT_CHARS + 1), encoding="utf-8")
    with pytest.raises(ValueError, match="потолке"):
        ROLE_SPEC.parse(str(path))


def test_an_empty_setting_is_legitimate():
    """★Отрицательный контроль: пустое значение — умолчание, а не забытая настройка."""
    assert ROLE_SPEC.parse("") == ""
    assert REPLY_SPEC.parse("") == ""


def test_a_proper_file_is_accepted(tmp_path):
    """★И второй отрицательный контроль: проверка, отвергающая всё, бесполезна."""
    path = tmp_path / "роль.txt"
    path.write_text("Ты — аналитик. Отвечай сухо.", encoding="utf-8")
    assert ROLE_SPEC.parse(str(path)) == str(path)


# ── текст заказчика доходит, наш вытесняется ───────────────────────────────


def test_the_customer_text_reaches_the_prompt(role_file):
    role_file("Ты — старший аналитик банка. Обращайся на «вы», без вводных слов.")
    built = prompts.system_prompt()

    assert "Обращайся на «вы»" in built
    # Наша роль именно ЗАМЕНЕНА, а не дополнена: иначе заказчик получил бы два
    # описания роли сразу, и модель выбирала бы между ними.
    assert "Upstream: разведка" not in built


def test_without_a_file_our_own_role_is_used():
    """Умолчание работает: система пригодна из коробки, без редакционной работы."""
    built = prompts.system_prompt()
    assert "Upstream: разведка" in built


# ── ГЛАВНОЕ: обязательные правила неотчуждаемы ─────────────────────────────


def test_mandatory_rules_survive_a_role_file_without_them(role_file):
    """★Файл заказчика без единого обязательного правила — они всё равно в промпте.

    Это и есть проверка того, что защита сделана УСТРОЙСТВОМ. Проверка вхождения
    обязательных фраз в файл была бы слабее вдвойне: она ловит удаление правила
    и НЕ ловит его переписывание в противоположное, а заодно запрещает заказчику
    менять то, что он вправе менять.
    """
    role_file("Ты — аналитик. Пиши коротко.")
    built = prompts.system_prompt()

    for mark in MANDATORY_MARKS:
        assert mark in built, f"обязательное правило пропало: {mark}"


def test_mandatory_rules_are_present_by_default():
    """Отрицательный контроль к предыдущему: без файла они тоже на месте.

    Без него тест выше прошёл бы и на системе, где обязательные правила
    приклеиваются, но пусты.
    """
    built = prompts.system_prompt()
    for mark in MANDATORY_MARKS:
        assert mark in built


def test_mandatory_rules_come_last(role_file):
    """Порядок несущий: конец инструкции модель держит сильнее."""
    role_file("Ты — аналитик. Пиши коротко.")
    built = prompts.system_prompt()

    assert built.index("Пиши коротко") < built.index("Никогда не выдумывай числа")


# ── реплика отказа ─────────────────────────────────────────────────────────


def test_the_refusal_reply_can_be_replaced(tmp_path, setting):
    path = tmp_path / "отказ.txt"
    path.write_text("Извините, это не мой профиль.", encoding="utf-8")
    setting("out_of_scope_file", str(path))

    assert prompts.out_of_scope_reply() == "Извините, это не мой профиль."


def test_the_refusal_reply_defaults_to_ours():
    assert "вне моей компетенции" in prompts.out_of_scope_reply()


def test_the_refusal_node_uses_the_customer_reply(tmp_path, setting):
    """Замена доходит до узла графа, а не остаётся в модуле промптов.

    ★Проверка через узел, а не через константу: подстановка на импорте закрепила
    бы то значение, какое было в момент загрузки модуля, и правка настройки
    молча не действовала бы.
    """
    from neftegaz.agent.graph import node_out_of_scope

    path = tmp_path / "отказ.txt"
    path.write_text("Не мой профиль.", encoding="utf-8")
    setting("out_of_scope_file", str(path))

    state = node_out_of_scope({"question": "как приготовить борщ"})
    assert state["answer"] == "Не мой профиль."
    assert state["history"][0]["answer"] == "Не мой профиль."


def test_a_file_that_disappeared_after_the_check_is_named_not_swallowed(tmp_path, setting):
    """Файл исчез между проверкой при старте и чтением — беда называется вслух.

    Тихий возврат к умолчанию дал бы работающую систему с чужим тоном.
    """
    setting("out_of_scope_file", str(tmp_path / "исчез.txt"))
    with pytest.raises(RuntimeError, match="не читается файл текста"):
        prompts.out_of_scope_reply()
