"""Отметка о том, ПО КАКИМ ПРАВИЛАМ собран индекс, и сверка её с настройкой.

★ЗАЧЕМ. Часть настроек описывает не поведение, а САМ ИНДЕКС: модель эмбеддингов,
размер фрагмента, перекрытие, имя коллекции. Поменяв их, вы не меняете ответы —
вы делаете индекс несогласованным с настройкой. Система при этом продолжит
отвечать уверенно, но из индекса, собранного по другим правилам: другой моделью
посчитанные векторы, другой длины фрагменты, а числа в ответе будут выглядеть
ровно так же. Это тот же класс дефекта, что молчаливый отказ поиска, — только
ещё тише, потому что никто не отказывал.

Поэтому при сборке индекса правила записываются рядом с ним, а интерфейс
сравнивает их с текущей настройкой и показывает расхождение, пока оно есть.
Метка не гаснет сама: она гаснет пересборкой.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from neftegaz.config import settings

__all__ = ["STAMPED_FIELDS", "mismatches", "read_stamp", "stamp_path", "write_stamp"]

# Настройки, меняющие сам индекс. ⚠Поле, забытое здесь, будет молча меняться
# без предупреждения — список важнее любой из его строк.
STAMPED_FIELDS = ("embedding_model", "chunk_size", "chunk_overlap", "collection")


def stamp_path() -> Path:
    """Отметка лежит рядом с хранилищем, а не внутри него.

    Внутри каталога Qdrant посторонний файл — чужая территория: движок вправе
    считать его своим или снести при пересоздании коллекции.
    """
    return Path(settings.qdrant_path).parent / "index-settings.json"


def write_stamp(chunks: int) -> Path:
    """Записать правила, по которым только что собран индекс."""
    payload = {field: getattr(settings, field) for field in STAMPED_FIELDS}
    payload["built_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    payload["chunks"] = chunks
    path = stamp_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_stamp() -> dict | None:
    """Правила прошлой сборки — None, если отметки нет.

    Отсутствие отметки НЕ равно согласию: индекс мог быть собран версией без
    отметки. Об этом честно говорит вызывающий, а не мы за него.
    """
    path = stamp_path()
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def mismatches() -> list[tuple[str, object, object]]:
    """Поля, где текущая настройка разошлась с той, по которой собран индекс.

    Пустой список при отсутствующей отметке: сказать «расхождений нет» там, где
    сравнивать не с чем, — то же самое «я не смог», выданное за «всё в порядке».
    Отличить одно от другого позволяет `read_stamp() is None`.
    """
    stamp = read_stamp()
    if stamp is None:
        return []
    return [
        (field, stamp.get(field), getattr(settings, field))
        for field in STAMPED_FIELDS
        if field in stamp and stamp.get(field) != getattr(settings, field)
    ]
