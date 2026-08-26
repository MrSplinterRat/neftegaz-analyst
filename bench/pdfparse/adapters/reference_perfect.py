"""ПОЛОЖИТЕЛЬНЫЙ контроль: стенд обязан уметь сказать «чисто».

Без него зелёная строка ничего не значила бы с другой стороны: стенд, который
всегда находит ошибку, так же бесполезен, как стенд, который всегда молчит.
"""
from __future__ import annotations

from adapters._perfect import tables

NAME = "контроль: идеал"
SABOTEUR = False


def parse(path: str) -> list[dict]:
    return tables()
