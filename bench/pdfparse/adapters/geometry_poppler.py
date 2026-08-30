"""КАНДИДАТ: координаты слов + привязка к шапке. Движок координат — poppler.

Рецепт пришёл от разведки (`_geometry.py`, он же /home/claude/tasks/pdf-recon/
eia_table_rows.py) и проверяется здесь ВТОРЫМ СЛОЕМ: эталон стенда снят с
отрисованных страниц независимо от того, чем мерила разведка.

Движок выбран poppler намеренно: `/bin/pdftotext` уже стоит в системе, значит
в образ едет ~5 МБ бинарника вместо 57 МБ колеса, и на диск рабочей станции
ничего ставить не надо. Разведка утверждает, что три движка координат дают
побайтово одинаковый результат — это утверждение стенд как раз и проверяет,
рядом лежит адаптер на pdfplumber.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

NAME = "геометрия (poppler)"
SABOTEUR = False
ENGINE = "poppler"


def parse(path: str) -> list[dict]:
    from _geometry import parse as geom

    tables = []
    for table in geom(path, ENGINE):
        columns = table["columns"]
        rows = []
        for row in table["rows"]:
            # ★Разрежённые ячейки уплотняются ПО ИНДЕКСУ КОЛОНКИ, пропуски —
            # пустой строкой. Складывать значения подряд нельзя: строка, где
            # заполнено не всё, уехала бы влево, и это ровно тот молчаливый
            # сдвиг, который стенд ловит отдельным вредителем.
            dense = [""] * len(columns)
            for index, value in row["cells"]:
                if 0 <= index < len(dense):
                    dense[index] = value
            rows.append({"label": row["label"], "values": dense})
        tables.append({"caption": table["caption"] or "", "columns": columns, "rows": rows})
    return tables
