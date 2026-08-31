"""Интерфейс не должен обращаться к посторонним серверам при запуске.

Повод у этих тестов конкретный. Streamlit на каждом безголовом старте делает
блокирующий HTTP-запрос к `checkip.amazonaws.com`, чтобы напечатать строку
«External URL» с внешним адресом машины. Настройками это не отключается.
Собственный вход `scripts/run_ui.py` эту возможность у библиотеки отбирает.
Здесь проверяется, что отбирает по-прежнему.

★Устройство проверки. Прибором служит подменённая функция, которая ходит в
сеть, — `net_util._make_blocking_http_get`. Она не выполняется, а запоминает
вызов. Если нейтрализация исчезнет, `get_external_ip()` до неё дойдёт, вызов
запишется, и тест упадёт.

★Отрицательный контроль встроен в набор отдельным тестом
(`test_probe_detects_leak_when_neutralisation_absent`): он снимает
нейтрализацию и требует, чтобы прибор течь УВИДЕЛ. Без него первый тест был бы
вечно-зелёным — он проходил бы и в случае, когда прибор вообще ничего не
измеряет.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# scripts/ не пакет и в sys.path не входит: там лежат исполняемые сценарии,
# а не библиотека. Для теста путь добавляется явно.
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_ui  # noqa: E402


@pytest.fixture
def network_probe(monkeypatch):
    """Прибор: считает попытки Streamlit сходить за внешним адресом.

    Заодно возвращает кэш внешнего адреса в исходное состояние. Без сброса
    тест ничего бы не проверял: значение, выставленное предыдущим тестом или
    самим импортом, закоротило бы проверку и она была бы зелёной всегда.
    """
    from streamlit import net_util

    calls: list[str] = []

    def _record(url: str, *_args, **_kwargs) -> None:
        calls.append(url)
        # Ничего не возвращаем: для Streamlit это «адрес выяснить не удалось»,
        # и он идёт дальше своей штатной веткой. В сеть при этом не ходим.

    monkeypatch.setattr(net_util, "_make_blocking_http_get", _record)
    monkeypatch.setattr(net_util, "_external_ip", None, raising=False)
    return calls


def test_entry_point_neutralises_external_ip_lookup(network_probe):
    """После нашего входа внешний адрес не запрашивается и не определён."""
    from streamlit import net_util

    run_ui._silence_external_ip_lookup()

    value = net_util.get_external_ip()

    # Несущее утверждение стоит первым: если проверка упадёт, в сообщении
    # должно быть названо то, что действительно случилось, — обращение наружу,
    # а не расхождение в типе возвращённого значения.
    assert network_probe == [], (
        f"в сеть всё-таки сходили: {network_probe}. "
        "Нейтрализация в scripts/run_ui.py перестала работать"
    )
    assert not value, (
        f"внешний адрес определён ({value!r}) — значит, строка «External URL» "
        "снова будет напечатана"
    )
    assert value == "", (
        "ожидалась пустая строка: она не-None (поэтому кэш считается заполненным "
        f"и запрос не выполняется) и при этом ложна. Получено: {value!r}"
    )


def test_probe_detects_leak_when_neutralisation_absent(network_probe):
    """Отрицательный контроль: без нейтрализации прибор течь ВИДИТ.

    Этот тест — не про продукт, а про сам прибор. Он падает, если проверка
    выше выродилась в проверку, которая проходит всегда.
    """
    from streamlit import net_util

    # Нейтрализация НЕ применяется — воспроизводим штатное поведение Streamlit.
    net_util.get_external_ip()

    assert network_probe, (
        "прибор не заметил обращения в сеть даже там, где оно заведомо "
        "происходит — значит, тест выше ничего не проверяет"
    )
    assert any("checkip.amazonaws.com" in url for url in network_probe), (
        f"ожидалось обращение к checkip.amazonaws.com, получено: {network_probe}"
    )


@pytest.mark.usefixtures("network_probe")
def test_websocket_origin_check_survives_neutralisation():
    """Пустой кэш не должен расширять список разрешённых источников.

    Тот же кэш читает проверка источника веб-сокета. Пустая строка попадает в
    список сравниваемых значений, и надо убедиться, что совпасть с ней ничто
    не может: иначе, отключив запрос в сеть, мы бы заодно открыли соединение
    кому попало.
    """
    from streamlit.web.server import server_util

    run_ui._silence_external_ip_lookup()

    assert server_util.is_url_from_allowed_origins("http://localhost") is True
    assert server_util.is_url_from_allowed_origins("http://evil.example.com") is False


# ── Переменные окружения: мост, которого нет у собственного входа даром ──────


def test_env_bridge_reads_streamlit_variables(monkeypatch):
    """`STREAMLIT_*` должны действовать и при запуске нашим входом.

    Штатный `streamlit` получает их через click. Наш вход click не поднимает,
    поэтому мост написан руками — и может быть потерян при правке. Образ
    полагается на `STREAMLIT_SERVER_ADDRESS`, поэтому потеря была бы поломкой
    запуска у заказчика, притом молчаливой.
    """
    monkeypatch.setenv("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")
    monkeypatch.setenv("STREAMLIT_SERVER_PORT", "8599")

    options = run_ui._config_options_from_env()

    assert options["server.address"] == "127.0.0.1"
    # Тип важен: порт — целое, а из окружения приходит строка.
    assert options["server.port"] == 8599


def test_env_bridge_ignores_unset_variables(monkeypatch):
    """Незаданная переменная не должна превращаться в настройку."""
    monkeypatch.delenv("STREAMLIT_SERVER_ADDRESS", raising=False)

    assert "server.address" not in run_ui._config_options_from_env()


# ── Образ: то, что было потеряно один раз и может потеряться снова ──────────


def _dockerfile() -> str:
    return (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_dockerfile_runs_our_entry_point():
    """CMD образа обязан звать наш вход, а не `streamlit run`."""
    text = _dockerfile()

    assert 'CMD ["python", "scripts/run_ui.py"]' in text, (
        "CMD образа больше не зовёт scripts/run_ui.py — значит, в контейнере "
        "снова работает штатный запуск Streamlit со всеми его обращениями наружу"
    )


def test_dockerfile_copies_streamlit_config():
    """Настройки интерфейса обязаны доезжать до образа.

    Однажды строки COPY здесь не было, и `toolbarMode = "viewer"` действовал
    только у разработчика: у заказчика в интерфейсе оставалась кнопка «Deploy»,
    ведущая на чужой публичный хостинг.
    """
    # `--chown=...` необязателен: он про владельца файлов, а не про то,
    # доезжают ли настройки. Проверяется именно доставка.
    pattern = r"^COPY (?:--chown=\S+ )?\.streamlit/ \./\.streamlit/$"
    assert re.search(pattern, _dockerfile(), re.M), (
        "в Dockerfile нет COPY .streamlit/ — настройки интерфейса не доедут до образа"
    )


def test_streamlit_config_hides_deploy_button():
    config = (PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")

    assert 'toolbarMode = "viewer"' in config
    assert "gatherUsageStats = false" in config


def test_dockerfile_baked_model_matches_config_default():
    """Имя модели в Dockerfile обязано совпадать с умолчанием в коде.

    В Dockerfile оно записано отдельной строкой (`ARG EMBEDDING_MODEL`), потому
    что модель запекается в образ раньше, чем туда попадают исходники.
    Дубликат сам о расхождении не сообщает — сообщает этот тест. Разошлись бы
    они тихо и дорого: в образ запеклась бы одна модель, а приложение при
    запуске полезло бы в интернет за другой.
    """
    from neftegaz.config import Settings

    match = re.search(r"^ARG EMBEDDING_MODEL=(\S+)$", _dockerfile(), re.M)
    assert match, "в Dockerfile нет строки ARG EMBEDDING_MODEL"

    assert match.group(1) == Settings().embedding_model, (
        "модель, запекаемая в образ, разошлась с умолчанием в neftegaz/config.py: "
        f"Dockerfile={match.group(1)!r}, config={Settings().embedding_model!r}"
    )


def test_baked_model_is_outside_the_mounted_volume():
    """Кэш модели не должен лежать под /app/data.

    /app/data при запуске закрывается томом хоста, и всё запечённое под этим
    путём становится недоступно — молча, с повторной загрузкой из интернета.
    """
    match = re.search(r"FASTEMBED_CACHE_PATH=(\S+)", _dockerfile())
    assert match, "в Dockerfile не задан FASTEMBED_CACHE_PATH"

    path = match.group(1)
    assert not path.startswith("/app/data"), (
        f"кэш модели ({path}) лежит под смонтированным томом — "
        "запекание в образ обессмысливается"
    )
