"""Готовность системы: проверка, которая умеет ответить «нет».

★ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ, ЕСЛИ КОНТЕЙНЕР И ТАК ПРОВЕРЯЛСЯ.

Проверялся, но не то. Прежняя проверка живости открывала страницу Streamlit, и
этого хватало, чтобы объявить контейнер здоровым. 26.08.2026 контейнер поднялся
за десять секунд в состоянии «здоров» с чистым журналом — и не работал вовсе:
кэш модели эмбеддингов метил в домашний каталог, а файловая система была
смонтирована только для чтения. Отказ вскрывался при первом же вопросе
пользователя, то есть тогда, когда проверять было уже поздно.

★Вопрос, который эту проверку осуждает: «что она скажет, если проверяемое
откажет ПОЛНОСТЬЮ?» Ответ был — «здоров». Значит проверки не было: она отвечала
на вопрос «поднялся ли веб-сервер», а читалась как ответ на вопрос «работает ли
система».

Работа стои́т на двух узлах, и оба надо трогать по-настоящему:

* **хранилище** — встроенный Qdrant пускает одного писателя, и каталог может
  быть занят другим процессом, отсутствовать или быть недоступным на запись;
* **эмбеддер** — модель весит 241 МБ, кладётся в кэш и при первом запуске может
  качаться из сети; именно на нём и произошёл отказ.

★ПРОВЕРКА ЗОВЁТ ПРОВЕРЯЕМОЕ, А НЕ ПОВТОРЯЕТ ЕГО. Здесь нет ни своего клиента
Qdrant, ни своего вызова fastembed: берётся тот же ``get_store()``, что и в
ответе на вопрос пользователя, с теми же настройками и тем же кэшем. Копия
тракта отстала бы от него молча — она не подаёт признаков устаревания.

★ПОЧЕМУ РЕЗУЛЬТАТ ПИШЕТСЯ В ФАЙЛ, А НЕ СЧИТАЕТСЯ ПО ЗАПРОСУ. Загрузка модели
дорога, а проверка живости идёт раз в полминуты; считать её каждый раз значило
бы держать 241 МБ ради ответа «да». Поэтому дорогую работу делает ОДИН РАЗ сам
процесс приложения — тот, который потом этой моделью и пользуется, — и
оставляет отметку. Дешёвая частая проверка читает отметку и страницу.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from neftegaz.config import settings

__all__ = [
    "Readiness",
    "marker_path",
    "probe",
    "read_marker",
    "run_probe_and_record",
]

# Проба эмбеддера идёт на короткой строке: проверяется, что модель загрузилась
# и считает, а не что она считает хорошо — качество проверяется не здесь.
PROBE_TEXT = "проверка готовности"


@dataclass(frozen=True)
class Readiness:
    """Исход проверки: годен ли каждый узел и почему нет.

    ``ok`` — оба узла живы. Отдельные поля нужны потому, что «не работает» без
    указания места не помогает: администратор контейнера должен видеть, чинить
    ему том с данными или кэш модели.
    """

    ok: bool
    storage_ok: bool
    embedder_ok: bool
    chunks: int
    storage_error: str
    embedder_error: str
    checked_at: float

    def as_line(self) -> str:
        """Одна строка для журнала контейнера."""
        if self.ok:
            return f"готовность: хранилище открыто ({self.chunks} фрагментов), эмбеддер загружен"
        parts = []
        if not self.storage_ok:
            parts.append(f"хранилище: {self.storage_error}")
        if not self.embedder_ok:
            parts.append(f"эмбеддер: {self.embedder_error}")
        return "готовность НЕ достигнута — " + "; ".join(parts)


def marker_path() -> Path:
    """Файл отметки. Лежит рядом с данными, а не в /tmp.

    ★Каталог данных — единственное место, про которое мы знаем, что оно
    доступно на запись: том монтируется туда. Отметка в /tmp пережила бы
    перезапуск процесса внутри того же контейнера и соврала бы о состоянии,
    которого больше нет.
    """
    return Path(settings.qdrant_path).parent / ".readiness.json"


def probe() -> Readiness:
    """Потрогать оба узла по-настоящему и вернуть исход.

    Исключения не выпускаются наружу: неготовность — это ответ, а не сбой
    программы. Проверка, падающая с трассировкой, неотличима от сломанной
    проверки.
    """
    storage_ok = embedder_ok = False
    storage_error = embedder_error = ""
    chunks = 0

    try:
        from neftegaz.rag.store import get_store

        store = get_store()
    except Exception as exc:  # noqa: BLE001 — причина сообщается, а не прячется
        message = f"{type(exc).__name__}: {exc}"
        return Readiness(False, False, False, 0, message, message, time.time())

    try:
        chunks = store.count()
        storage_ok = True
    except Exception as exc:  # noqa: BLE001
        storage_error = f"{type(exc).__name__}: {exc}"

    try:
        # ★Именно вычисление вектора, а не создание объекта: fastembed
        # откладывает загрузку весов до первого счёта, и «эмбеддер создан»
        # ничего не говорит о том, доступен ли кэш на запись — а падало ровно
        # это.
        vector = store._embed_query(PROBE_TEXT)  # noqa: SLF001
        if not vector:
            raise ValueError("эмбеддер вернул пустой вектор")
        embedder_ok = True
    except Exception as exc:  # noqa: BLE001
        embedder_error = f"{type(exc).__name__}: {exc}"

    return Readiness(
        ok=storage_ok and embedder_ok,
        storage_ok=storage_ok,
        embedder_ok=embedder_ok,
        chunks=chunks,
        storage_error=storage_error,
        embedder_error=embedder_error,
        checked_at=time.time(),
    )


def run_probe_and_record(path: Path | None = None) -> Readiness:
    """Проверить узлы и записать отметку. Возвращает тот же исход.

    Запись атомарная — через временный файл и переименование. Проверка живости
    читает эту отметку параллельно, и застать её на середине записи значило бы
    получить «не готов» на исправной системе, то есть шум вместо сигнала.
    """
    target = path or marker_path()
    result = probe()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(result), ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, target)
    except OSError:
        # ★Не записалась отметка — не повод объявлять систему неисправной, но и
        # не повод молчать: проверка живости не найдёт файла и скажет «не
        # готов». Это верный ответ: система, о состоянии которой нельзя
        # узнать, не считается работающей.
        pass
    return result


def read_marker(path: Path | None = None) -> Readiness | None:
    """Прочитать отметку. ``None`` — отметки нет или она нечитаема.

    ★Нечитаемая отметка приравнивается к отсутствующей, а не к «здоров».
    Обратное решение превратило бы порчу файла в зелёный свет.
    """
    target = path or marker_path()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        return Readiness(
            ok=bool(raw["ok"]),
            storage_ok=bool(raw["storage_ok"]),
            embedder_ok=bool(raw["embedder_ok"]),
            chunks=int(raw["chunks"]),
            storage_error=str(raw["storage_error"]),
            embedder_error=str(raw["embedder_error"]),
            checked_at=float(raw["checked_at"]),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
