"""ВРЕДИТЕЛЬ: отрезает последний столбец — ровно дефект, найденный 26.08.2026.

Шаблон строки требовал пробела после каждого значения, а последнее кончается
переводом строки. Так у 903 строк из 938 пропадала годовая колонка, то есть
прогноз на дальний год — тот самый, о котором спрашивают.
"""
from __future__ import annotations

from adapters._perfect import tables

NAME = "вредитель: минус столбец"
SABOTEUR = True


def parse(path: str) -> list[dict]:  # noqa: ARG001 — сигнатуру задаёт интерфейс адаптера; данные синтетические, с диска не читаются
    result = tables()
    for table in result:
        for row in table["rows"]:
            row["values"] = row["values"][:-1]
    return result
