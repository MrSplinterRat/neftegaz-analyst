"""КАНДИДАТ: exlayout — структурный разбор PDF на Rust (/opt/eidos/exlayout).

Стои́т в стенде рядом с `library_pdf2xml` и `geometry_poppler` как ПРИЁМКА
ЗАМЕНЫ: exlayout предполагается поставить вместо pdf2xml, и утверждать
равноценность без замера значило бы верить себе на слово.

Как он отвечает. Бинарь печатает одну строку JSON на файл:

    exlayout --quiet --cells ФАЙЛ

Схема `exlayout.table-cells`: документ → таблицы → строки → ячейки. Таблица там
— не «таблица со страницы», а БЛОК: прогон строк, занимающих достаточно колонок
(exlayout режет страницу на блоки и о каждом судит отдельно). Одна печатная
таблица отчёта EIA распадается на несколько блоков, а строки, не дотянувшие до
порога, в блок не входят и приезжают отдельным полем `above` — «что разрез
срезал над блоком». Куски, не легшие на сетку вовсе (заголовок таблицы, ПОЛОСА
ГОДОВ над кварталами), приезжают в `above_loose` со своими координатами.

Вся работа этого адаптера — собрать из блоков одной страницы то, что человек
видит одной таблицей. Три правила, и каждое опирается на данные, а не на
порядок слов:

1. ГРАНИЦА ТАБЛИЦЫ — поток содержимого (`content_object`). У EIA STEO это ровно
   страница.
2. ШАПКА НИЖНЕГО ЯРУСА — строка из `above`, где больше половины ячеек читаются
   как «Q1»…«Q4».
3. ШАПКА ВЕРХНЕГО ЯРУСА — куски `above_loose`, стоящие на одной высоте, среди
   которых не меньше двух четырёхзначных годов. Год центрирован над своей
   четвёркой кварталов и потому не принадлежит ни одной колонке — exlayout
   отказывается назначить ему номер и отдаёт координату. Колонка относится к
   тому году, чей центр к ней ближе (границы — середины между соседними
   годами). Это и есть связь «какой это Q1», которой в плоском тексте нет.

★Значения отдаются ДОСЛОВНО, строками, вместе с прочерками. Пропуск колонки
заполняется пустой строкой, а не выбрасывается: выброшенный прочерк уводит
весь остаток ряда влево, оставаясь правдоподобным на вид.
"""

from __future__ import annotations

import json
import re
import subprocess

NAME = "exlayout (Rust)"
SABOTEUR = False

BINARY = "/opt/eidos/exlayout/target/release/exlayout"
TIMEOUT = 300

# ★СОБИРАТЬ ЛИ СТРОКИ ИЗ КУСКОВ, НЕ ЛЁГШИХ НА СЕТКУ.
#
# Выключатель стои́т здесь, а не спрятан в коде, потому что он меняет СМЫСЛ
# замера. С ним стенд судит «exlayout плюс проекция на его же полосы», без
# него — то, что exlayout положил на сетку сам. Второе честнее как оценка
# движка, первое ближе к тому, что получит потребитель.
#
# Что именно делает проекция: куски с ОДИНАКОВОЙ высотой `y` собираются в
# строку, и каждый ложится в ту колонку, в чью полосу попал его центр. Полосы
# и высоты — из ответа exlayout; ни новой геометрии, ни кластеризации здесь
# нет. Нужна она потому, что строчная ось иногда отвергает настоящую строку
# данных: на `EIA_STEO_2026-07`, Table 3c, так вышло с «World total» и со всей
# полосой кварталов.
RECOVER_LOOSE_ROWS = True

YEAR = re.compile(r"^(19|20)\d\d$")
QUARTER = re.compile(r"^Q[1-4]$", re.IGNORECASE)
CAPTION = re.compile(r"^\s*Table\s+\S", re.IGNORECASE)


def _run(path: str) -> dict:
    """Позвать бинарь и вернуть разобранный ответ."""
    done = subprocess.run(
        [BINARY, "--quiet", "--cells", path],
        capture_output=True,
        timeout=TIMEOUT,
        check=False,
    )
    # Код 2 значит «инструмент не смог» (нет файла, негодные аргументы);
    # отказ по документу — это код 0 и законный отчёт с outcome.
    if done.returncode == 2:
        raise RuntimeError(done.stderr.decode("utf-8", "replace")[:400])
    return json.loads(done.stdout.decode("utf-8"))


def _dense(cells: list[dict], width: int) -> list[str]:
    """Ячейки строки в плотный список по номеру колонки, пропуск — пустая строка."""
    out = [""] * width
    for cell in cells:
        col = cell.get("col")
        if isinstance(col, int) and 0 <= col < width:
            out[col] = str(cell.get("text", ""))
    return out


def _loose_rows(block: dict) -> list[dict]:
    """Строки, собранные из кусков одной высоты проекцией на полосы колонок.

    Возвращает записи той же формы, что и `above`: {"page_row": …, "cell": […]}.
    Колонка назначается вхождением ЦЕНТРА куска в полосу; кусок, не попавший ни
    в одну, отбрасывается — приписать его ближайшей значило бы придумать место.
    """
    if not RECOVER_LOOSE_ROWS:
        return []
    bands = block.get("column") or []
    if not bands:
        return []
    by_height: dict[float, list[dict]] = {}
    for piece in block.get("above_loose") or []:
        y = piece.get("y")
        if isinstance(y, (int, float)):
            by_height.setdefault(round(float(y), 3), []).append(piece)

    out: list[dict] = []
    for y in sorted(by_height, reverse=True):  # ось y растёт вверх: сверху вниз
        cells = []
        for piece in by_height[y]:
            centre = (float(piece.get("x0", 0.0)) + float(piece.get("x1", 0.0))) / 2
            for col, band in enumerate(bands):
                if float(band.get("x0", 0.0)) <= centre <= float(band.get("x1", 0.0)):
                    cells.append({"col": col, "text": str(piece.get("text", ""))})
                    break
        if len(cells) >= 3:
            out.append({"page_row": -1, "y": y, "cell": sorted(cells, key=lambda c: c["col"])})
    return out


def _caption(blocks: list[dict]) -> str:
    """Заголовок таблицы: первая строка вида «Table …» среди не легшего.

    ★Пробелы сжимаются. Заголовок в PDF набран с двойным пробелом после номера
    («Table 9a.  U.S. …» — так его отдают и exlayout, и mutool), а стенд ищет
    вхождение первых сорока знаков эталона, где пробел один. Это разница
    НАБОРА, а не чтения, и снимать её — работа адаптера.
    """
    for block in blocks:
        for piece in block.get("above_loose") or []:
            text = " ".join(str(piece.get("text", "")).split())
            if CAPTION.match(text):
                return text
    for block in blocks:
        for piece in block.get("above_loose") or []:
            text = " ".join(str(piece.get("text", "")).split())
            if text:
                return text
    return ""


def _lower_tier(blocks: list[dict], width: int) -> tuple[list[str], list[dict], dict | None]:
    """Нижний ярус шапки: строка `above`, где преобладают «Q1»…«Q4».

    Возвращает подписи по номеру колонки блока, саму строку (нужна, чтобы не
    считать её потом строкой данных) и БЛОК, в котором она нашлась — полосы
    колонок надо брать у него, а не у первого попавшегося.
    """
    best: tuple[int, list[str], dict, dict] | None = None
    for block in blocks:
        for row in (block.get("above") or []) + _loose_rows(block):
            texts = [str(c.get("text", "")).strip() for c in row.get("cell") or []]
            hits = sum(1 for t in texts if QUARTER.match(t))
            if hits * 2 > max(len(texts), 1) and (best is None or hits > best[0]):
                best = (hits, _dense(row.get("cell") or [], width), row, block)
    if best is None:
        return [], [], None
    return best[1], [best[2]], best[3]


def _upper_tier(blocks: list[dict], below: float | None) -> list[dict]:
    """Верхний ярус: куски одной высоты НАД нижним ярусом, где не меньше двух годов.

    ★Условие «над» несущее. Без него верхним ярусом объявляется сам нижний:
    в шапке EIA последние три колонки подписаны годами («2025 2026 2027» под
    «Year»), и полоса кварталов проходит проверку «не меньше двух годов» ничуть
    не хуже полосы годов — да ещё и длиннее её вчетверо. Замер на Table 3c:
    без условия выходило «Q1» вместо «2025Q1» и «20252025» вместо «2025».
    Ось `y` растёт вверх, поэтому «над» значит БОЛЬШЕ.
    """
    by_height: dict[float, list[dict]] = {}
    for block in blocks:
        for piece in block.get("above_loose") or []:
            y = piece.get("y")
            if isinstance(y, (int, float)) and (below is None or float(y) > below + 0.5):
                by_height.setdefault(round(float(y), 2), []).append(piece)
    best: list[dict] = []
    best_y = None
    for y, pieces in by_height.items():
        years = sum(1 for p in pieces if YEAR.match(str(p.get("text", "")).strip()))
        # Из подходящих берётся БЛИЖАЙШАЯ снизу полоса: ярус над ярусом, а не
        # любая строка с годами где-то выше по странице.
        if years >= 2 and (best_y is None or y < best_y):
            best, best_y = pieces, y
    return sorted(best, key=lambda p: float(p.get("x", 0.0)))


def _columns(blocks: list[dict], width: int) -> tuple[list[str], list[dict]]:
    """Шапка колонок. Двухъярусная, если верхний ярус нашёлся; иначе нижний."""
    lower, used, home = _lower_tier(blocks, width)
    if not lower:
        return [], used
    below = used[0].get("y") if used else None
    upper = _upper_tier(blocks, float(below) if isinstance(below, (int, float)) else None)
    bands = (home.get("column") or []) if home else []
    if not upper or not bands:
        return lower[1:], used

    # ★Центр берётся из РАЗМАХА куска (`x0`/`x1`), а не из левого края. Год
    # центрирован над своей четвёркой кварталов, и разница между краем и
    # центром — половина ширины подписи, около шести пунктов. По левым краям
    # колонка «2027Q4» уезжает в группу «Year»; по размахам стои́т на месте.
    marks = [
        ((float(p.get("x0", 0.0)) + float(p.get("x1", 0.0))) / 2, str(p.get("text", "")).strip())
        for p in upper
    ]
    marks.sort()
    edges = [(marks[i][0] + marks[i + 1][0]) / 2 for i in range(len(marks) - 1)]

    out: list[str] = []
    for col in range(1, width):
        band = bands[col] if col < len(bands) else None
        if band is None:
            out.append(lower[col])
            continue
        centre = (float(band.get("x0", 0.0)) + float(band.get("x1", 0.0))) / 2
        which = 0
        while which < len(edges) and centre > edges[which]:
            which += 1
        top = marks[which][1] if which < len(marks) else ""
        # Верхний ярус, не являющийся годом («Year» над годовыми колонками),
        # ничего к нижнему не добавляет: там уже стои́т сам год.
        out.append(f"{top}{lower[col]}" if YEAR.match(top) else lower[col])
    return out, used


def parse(path: str) -> list[dict]:
    answer = _run(path)
    groups: dict[object, list[dict]] = {}
    for table in answer.get("table") or []:
        groups.setdefault(table.get("content_object"), []).append(table)

    out: list[dict] = []
    for blocks in groups.values():
        blocks.sort(key=lambda b: b.get("id", 0))
        width = max((b.get("cols") or 0) for b in blocks)
        # Блоки с иной геометрией колонок — другая таблица на той же странице.
        blocks = [b for b in blocks if (b.get("cols") or 0) == width]
        if width < 2:
            continue
        columns, used = _columns(blocks, width)

        rows: list[dict] = []
        for block in blocks:
            # Сперва то, что разрез срезал НАД блоком (в отчётах EIA это строки
            # с подписью в две строки: имя сверху, единицы и числа снизу), потом
            # сам блок. Порядок повторяет порядок на странице.
            for row in (block.get("above") or []) + _loose_rows(block):
                if row in used:
                    continue
                cells = row.get("cell") or []
                if len(cells) < 3:
                    continue
                dense = _dense(cells, width)
                rows.append({"label": dense[0], "values": dense[1:]})
            for row in block.get("row") or []:
                dense = _dense(row.get("cell") or [], width)
                rows.append({"label": dense[0], "values": dense[1:]})

        out.append({"caption": _caption(blocks), "columns": columns, "rows": rows})
    return out
