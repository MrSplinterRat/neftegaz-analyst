"""Идеальный разбор, взятый прямо из эталона. Служебный, сам не запускается.

Нужен двум сортам контролей: положительному (стенд обязан уметь сказать «чисто»)
и вредителям (им есть что ломать). Настоящим кандидатом не является и никогда
им не станет: он не читает PDF вовсе.
"""

from __future__ import annotations

import json
import pathlib

TRUTH = json.loads(
    (pathlib.Path(__file__).resolve().parent.parent / "truth.json").read_text(encoding="utf-8")
)


def tables() -> list[dict]:
    return [
        {
            "caption": table["заголовок"],
            "columns": list(table["колонки"]),
            "rows": [
                {"label": row["подпись"], "values": list(row["значения"])}
                for row in table["строки"]
            ],
        }
        for table in TRUTH["таблицы"]
    ]
