"""ВРЕДИТЕЛЬ: сдвигает ряд на одну колонку.

Самый опасный сорт отказа: числа на месте, все до одного настоящие, ответ
выглядит обычным — и врёт. Стенд обязан не просто насчитать ошибки, а НАЗВАТЬ
это сдвигом, иначе пятнадцать одинаковых сообщений скроют одну причину.
"""

from __future__ import annotations

from adapters._perfect import tables

NAME = "вредитель: сдвиг на 1"
SABOTEUR = True


def parse(path: str) -> list[dict]:  # noqa: ARG001 — сигнатуру задаёт интерфейс адаптера; данные синтетические, с диска не читаются
    result = tables()
    for table in result:
        for row in table["rows"]:
            row["values"] = ["0.00"] + row["values"][:-1]
    return result
