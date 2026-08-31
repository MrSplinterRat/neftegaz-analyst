"""Правка параметров хода: проверка, применение и жизнь дольше процесса.

★ЧТО ЗДЕСЬ ГЛАВНОЕ. Настройка, которую можно поправить в интерфейсе, но которая
забывается при перезапуске, — не настройка, а прихоть вкладки. Поэтому половина
тестов проверяет не «значение изменилось», а «значение вернулось после того, как
конфигурация была прочитана заново».

★Настоящий перезапуск ПРОЦЕССА проверяется отдельным тестом, который запускает
python заново: перечитывание модуля внутри того же процесса разделяет с
проверяемым слишком много механизма, чтобы считаться доказательством.

★И столько же внимания — границе списка. Правимость параметров хода имеет смысл
ровно потому, что параметры КОРПУСА так править нельзя: их правка делает
собранный индекс несогласованным с настройкой. Список закрыт, и тест на это
падает, если кто-нибудь добавит в него размер фрагмента.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from neftegaz import config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Файл правок во временном каталоге — настоящий не трогаем."""
    path = tmp_path / "settings-overrides.json"
    monkeypatch.setattr(config, "overrides_path", lambda: path)
    before = {
        parameter.field: getattr(config.settings, parameter.field)
        for parameter in config.TURN_PARAMETERS
    }
    yield path
    for name, value in before.items():
        object.__setattr__(config.settings, name, value)


# ── проверка значения ──────────────────────────────────────────────────────


def test_a_value_out_of_range_is_refused_loudly(sandbox):
    with pytest.raises(ValueError, match="больше допустимого"):
        config.set_turn_parameter("top_k", 500)
    assert not sandbox.exists(), "отклонённое значение не должно попадать на диск"


@pytest.mark.usefixtures("sandbox")
def test_a_value_of_the_wrong_kind_is_refused_loudly():
    with pytest.raises(ValueError, match="целым числом"):
        config.set_turn_parameter("top_k", "пять")


@pytest.mark.usefixtures("sandbox")
def test_a_region_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="не подходит по форме"):
        config.set_turn_parameter("web_region", "Россия")


@pytest.mark.usefixtures("sandbox")
def test_a_parameter_outside_the_list_cannot_be_set_at_all():
    """★Граница списка и есть смысл разделения настроек."""
    with pytest.raises(KeyError):
        config.set_turn_parameter("chunk_size", 900)
    with pytest.raises(KeyError):
        config.set_turn_parameter("embedding_model", "что-нибудь другое")


def test_the_corpus_parameters_are_not_in_the_editable_list():
    editable = {parameter.field for parameter in config.TURN_PARAMETERS}
    for forbidden in ("chunk_size", "chunk_overlap", "embedding_model", "collection"):
        assert forbidden not in editable, (
            f"{forbidden} правится из интерфейса — это делает индекс несогласованным"
        )


# ── применение и хранение ──────────────────────────────────────────────────


def test_a_good_value_applies_at_once_and_is_written_down(sandbox):
    config.set_turn_parameter("top_k", "9")
    assert config.settings.top_k == 9, "правка обязана действовать со следующего вопроса"
    assert json.loads(sandbox.read_text(encoding="utf-8"))["top_k"] == 9


def test_reset_returns_the_value_from_env_and_forgets_the_override(sandbox):
    from_env = config.env_settings.top_k
    config.set_turn_parameter("top_k", from_env + 3)
    config.reset_turn_parameter("top_k")
    assert config.settings.top_k == from_env
    assert "top_k" not in json.loads(sandbox.read_text(encoding="utf-8"))


def test_setting_the_value_env_already_gives_is_not_recorded_as_a_change(sandbox):
    """★Предупреждение без повода обесценивает себя к тому дню, когда повод будет.

    Поймано приёмкой в браузере: панель написала «изменено из интерфейса:
    сейчас 5, а .env говорит 5».
    """
    config.set_turn_parameter("top_k", config.env_settings.top_k)
    assert json.loads(sandbox.read_text(encoding="utf-8")) == {}


@pytest.mark.usefixtures("sandbox")
def test_env_settings_keeps_saying_what_env_says():
    """Снимок `.env` не должен ехать вслед за правкой — на нём стои́т показ расхождения."""
    before = config.env_settings.top_k
    config.set_turn_parameter("top_k", before + 2)
    assert config.env_settings.top_k == before


def test_a_broken_overrides_file_does_not_stop_the_start(sandbox, capsys):
    sandbox.write_text("{это не json", encoding="utf-8")
    assert config.read_overrides() == {}
    assert "не прочитан" in capsys.readouterr().out


def test_an_unknown_name_in_the_file_is_skipped_out_loud(sandbox, capsys):
    sandbox.write_text(json.dumps({"chunk_size": 900}), encoding="utf-8")
    config._load_saved_overrides()
    assert "chunk_size" in capsys.readouterr().out


def test_an_unusable_value_in_the_file_is_skipped_out_loud(sandbox, capsys):
    sandbox.write_text(json.dumps({"top_k": 10_000}), encoding="utf-8")
    config._load_saved_overrides()
    printed = capsys.readouterr().out
    assert "top_k" in printed and "больше допустимого" in printed
    assert config.settings.top_k != 10_000, "непригодная правка не должна применяться"


# ── переживание перезапуска ПРОЦЕССА ───────────────────────────────────────


def test_an_override_survives_a_real_process_restart(tmp_path):
    """★Настоящий перезапуск, а не перечитывание модуля в том же процессе.

    Перечитывание разделяет с проверяемым и загруженные модули, и уже
    разобранное окружение; такая проверка ответила бы «пережило» и в случае,
    когда на диск не записалось ничего.
    """
    overrides = tmp_path / "settings-overrides.json"
    write = (
        "from neftegaz import config;"
        "config.set_turn_parameter('top_k', 11);"
        "print(config.settings.top_k)"
    )
    read = "from neftegaz.config import settings; print(settings.top_k)"
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(PROJECT_ROOT),
        "SETTINGS_OVERRIDES": str(overrides),
        # ★.env проекта из подпроцесса не читаем: тест обязан мерить наше
        # хранилище правок, а не настройку конкретной машины.
        "RAG_TOP_K": "5",
    }
    first = subprocess.run(
        [sys.executable, "-c", write], capture_output=True, text=True, env=env, check=True
    )
    assert first.stdout.strip() == "11"
    assert json.loads(overrides.read_text(encoding="utf-8"))["top_k"] == 11

    second = subprocess.run(
        [sys.executable, "-c", read], capture_output=True, text=True, env=env, check=True
    )
    assert second.stdout.strip() == "11", "правка не пережила перезапуск процесса"

    # Отрицательный контроль: без файла правок тот же процесс обязан вернуться
    # к значению из окружения. Иначе «пережило» ничего не доказывает — так
    # ответила бы и проверка, которая всегда печатает 11.
    overrides.unlink()
    third = subprocess.run(
        [sys.executable, "-c", read], capture_output=True, text=True, env=env, check=True
    )
    assert third.stdout.strip() == "5"
