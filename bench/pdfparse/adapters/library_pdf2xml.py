"""КАНДИДАТ: библиотека /opt/eidos/pdf2xml — та же геометрия, но упакованная.

Стоит рядом с `geometry_poppler` не ради второго мнения, а как ПРИЁМКА
УПАКОВКИ. Рецепт переезжал из разведочного скрипта в библиотеку с
переписыванием: появились типы, наследование заголовка пошло через смежность
страниц, полоса верхнего яруса перестала считаться строкой данных. Каждое из
этих изменений могло сдвинуть результат, и утверждать «просто переложил» без
замера значило бы верить себе на слово.

Оба адаптера меряются одним прибором. Расхождение между ними — дефект
упаковки, и стенд назовёт его сам.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/opt/eidos/pdf2xml")

NAME = "библиотека pdf2xml"
SABOTEUR = False


def parse(path: str) -> list[dict]:
    from pdf2xml import parse_pdf

    document = parse_pdf(path, keep_words=False)
    return [
        {
            "caption": table.caption,
            "columns": table.column_labels(),
            "rows": [
                {"label": row.label, "values": row.values(table.width)}
                for row in table.rows
            ],
        }
        for table in document.tables()
    ]
