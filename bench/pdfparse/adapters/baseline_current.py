"""ТЕКУЩИЙ КОНВЕЙЕР как кандидат: pypdf плюс наши регулярки.

Стоит на стенде не для красоты. Без него «кандидат восстановил обе строки»
осталось бы утверждением без величины: непонятно, во сколько раз это лучше
того, что уже работает, и стоит ли овчинка выделки. База обязана быть измерена
тем же прибором, что и замена.

Таблицы собираются ровно тем кодом, который сегодня строит индекс:
`caption_positions`, `column_header_after`, `table_rows`. Ничего специально для
стенда не улучшается — иначе стенд мерил бы не то, что работает в проде.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

NAME = "база: pypdf+регулярки"
SABOTEUR = False

_VALUE = re.compile(r"^(?:-?[\d,]+(?:\.\d+)?|-)$")


def parse(path: str) -> list[dict]:
    from neftegaz.rag.chunking import (
        _DOTS,
        caption_positions,
        column_header_after,
        table_rows,
    )
    from neftegaz.rag.ingest import read_pdf_pages

    stream = "".join(page["text"] for page in read_pdf_pages(path))
    captions = caption_positions(stream)
    rows = table_rows(stream)

    tables = []
    for index, (start, caption) in enumerate(captions):
        end = captions[index + 1][0] if index + 1 < len(captions) else len(stream)
        header = column_header_after(stream, start)
        collected = []
        for row_start, row_end in rows:
            if not (start <= row_start < end):
                continue
            text = stream[row_start:row_end]
            dots = _DOTS.search(text)
            label = text[: dots.start()] if dots else text
            tail = text[dots.end() :] if dots else ""
            collected.append(
                {
                    # Подпись чистится от затянувшейся предыдущей строки: граница
                    # подписи — последнее число выше, а у строки-раздела чисел нет.
                    "label": label.strip().split("\n")[-1].strip(),
                    "values": [t for t in tail.split() if _VALUE.match(t)],
                }
            )
        tables.append(
            {
                "caption": caption,
                "columns": header.split() if header else [],
                "rows": collected,
            }
        )
    return tables
