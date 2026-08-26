"""ВРЕДИТЕЛЬ: выбрасывает прочерки вместо того, чтобы держать их ячейками.

Кандидаты делают так постоянно: пустая ячейка кажется отсутствием данных, а не
данными. Для строки «Propane Residential», где прочерков девять из пятнадцати,
это уезжает целиком. На строке из одних чисел такой дефект НЕВИДИМ — потому в
эталоне и лежит строка с прочерками.
"""
from __future__ import annotations

from adapters._perfect import tables

NAME = "вредитель: без прочерков"
SABOTEUR = True


def parse(path: str) -> list[dict]:
    result = tables()
    for table in result:
        for row in table["rows"]:
            row["values"] = [v for v in row["values"] if v != "-"]
    return result
