#!/usr/bin/env python3
"""Строки таблиц EIA STEO со ЧТЕНИЕМ поля колонки, а не догадкой по расстоянию.

Структура берётся из КООРДИНАТ слов. Движок координат сменный, результат один
и тот же (проверено на 8 отчётах: 101043 ячейки, расхождений ноль):
    pdfplumber  page.extract_words()          MIT,  +57 МБ, 4.85 с/отчёт
    poppler     pdftotext -bbox-layout        GPL-бинарь, ~5 МБ, 0.16 с/отчёт
    pymupdf     page.get_text("words")        AGPL, +64 МБ, 0.28 с/отчёт

Рецепт:
  1. слова группируются в строки по y (допуск 1.6 pt);
  2. ШАПКА — строка, где >=8 токенов суть Q1..Q4 или четырёхзначный год;
  3. границы колонок = середины между x-центрами соседних слов шапки;
  4. ВЕРХНИЙ ЯРУС (полоса лет / "Year") привязывается к колонкам так же ->
     возвращаются склеенные обозначения 2025Q1 … 2027Q4 + 2025/2026/2027;
  5. каждое значение (число ИЛИ прочерк) попадает в колонку, чей интервал
     накрывает его x-центр; всё левее первой колонки — часть подписи;
  6. ЗАГОЛОВОК ТАБЛИЦЫ — строка "Table N." с наименьшим y НАД шапкой; если на
     странице-продолжении её нет, наследуется с предыдущей страницы.
     ★Именно это чинит порядок чтения: в потоке pypdf заголовок стоит ПОСЛЕ
     своих строк на 168 из 208 страниц-таблиц корпуса.

Запуск:  python3 eia_table_rows.py <файл.pdf> [--engine pdfplumber|poppler|pymupdf] [страница]
"""
# ruff: noqa: E701, E702, UP031
# ★Плотный стиль этого файла — осознанный и локальный: это standalone-скрипт
# разбора геометрии, где `if …: continue` и `a = x; break` держат шаг алгоритма
# на одной строке и читаются как псевдокод из докстринга выше. Проценты в печати
# оставлены ради выравнивания колонок (`%-46s`), которое f-строкой не короче.
# Запреты сняты ФАЙЛОМ, а не в настройках линтера: в остальном коде эти правила
# должны продолжать работать.
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

VALUE = re.compile(r'^(-|--|NA|W|-?\d[\d,]*\.\d+)$')
CAPTION = re.compile(r'^Table\s+\d+[a-z]?\.')
Y_TOL = 1.6
NS = "{http://www.w3.org/1999/xhtml}"


def words_pdfplumber(path):
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            yield [(w["text"], w["x0"], w["x1"], w["top"])
                   for w in p.extract_words(use_text_flow=False)]
            p.flush_cache()


def words_pymupdf(path):
    import pymupdf
    for p in pymupdf.open(path):
        yield [(w[4], w[0], w[2], w[1]) for w in p.get_text("words")]


def words_poppler(path):
    out = subprocess.run(["/bin/pdftotext", "-bbox-layout", path, "-"],
                         capture_output=True, check=True).stdout
    for pg in ET.fromstring(out).iter(NS + "page"):
        yield [(w.text or "", float(w.get("xMin")), float(w.get("xMax")), float(w.get("yMin")))
               for w in pg.iter(NS + "word")]


ENGINES = {"pdfplumber": words_pdfplumber, "pymupdf": words_pymupdf, "poppler": words_poppler}


def to_lines(ws, tol=Y_TOL):
    ws = sorted(ws, key=lambda w: (w[3], w[1]))
    out, cur, ctop = [], [], None
    for w in ws:
        if ctop is None or abs(w[3] - ctop) <= tol:
            cur.append(w); ctop = w[3] if ctop is None else ctop
        else:
            out.append(cur); cur = [w]; ctop = w[3]
    if cur: out.append(cur)
    return out


def parse_page(ws, inherited_caption=None):
    """-> (caption, columns, rows) либо (caption, None, []) если таблицы нет."""
    lines = to_lines(ws)
    caption = inherited_caption
    for L in lines:
        s = " ".join(w[0] for w in sorted(L, key=lambda w: w[1]))
        if CAPTION.match(s):
            caption = s; break
    hi = None
    for k, L in enumerate(lines):
        toks = [w[0] for w in L]
        q = sum(1 for t in toks if t in ("Q1", "Q2", "Q3", "Q4") or (len(t) == 4 and t.isdigit()))
        if q >= 8 and q >= len(toks) - 2:
            hi = k; break
    if hi is None:
        return caption, None, []
    hdr = sorted(lines[hi], key=lambda w: w[1])
    centers = [(w[1] + w[2]) / 2 for w in hdr]
    names = [w[0] for w in hdr]
    band = [w for w in (sorted(lines[hi - 1], key=lambda w: w[1]) if hi else [])
            if (len(w[0]) == 4 and w[0].isdigit()) or w[0] == "Year"]
    columns = []
    for c, n in zip(centers, names, strict=True):  # оба из одного hdr, длины равны по построению
        b = min(band, key=lambda w: abs((w[1] + w[2]) / 2 - c))[0] if band else ""
        columns.append(n if b in ("", "Year") else b + n)
    bounds = [(centers[i] + centers[i + 1]) / 2 for i in range(len(centers) - 1)]
    left = centers[0] - (bounds[0] - centers[0]) if bounds else centers[0]

    def col_of(x):
        i = 0
        while i < len(bounds) and x > bounds[i]:
            i += 1
        return i

    rows = []
    for k, L in enumerate(lines):
        if k == hi: continue
        L = sorted(L, key=lambda w: w[1])
        vals = [w for w in L if VALUE.match(w[0]) and (w[1] + w[2]) / 2 > left]
        if len(vals) < 3: continue
        ids = {id(w) for w in vals}
        label = " ".join(w[0] for w in L if id(w) not in ids).strip(" .")
        rows.append({"label": label,
                     "cells": [(col_of((w[1] + w[2]) / 2), w[0]) for w in vals]})
    return caption, columns, rows


def parse(path, engine="pdfplumber"):
    out = []
    cap = None
    for pno, ws in enumerate(ENGINES[engine](path), start=1):
        cap, cols, rows = parse_page(ws, cap)
        if cols:
            out.append({"page": pno, "caption": cap, "columns": cols, "rows": rows})
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    engine = "pdfplumber"
    if "--engine" in sys.argv:
        engine = sys.argv[sys.argv.index("--engine") + 1]
        args = [a for a in args if a != engine]
    only = int(args[1]) if len(args) > 1 else None
    for t in parse(args[0], engine):
        if only and t["page"] != only: continue
        print("=== стр %d | %s" % (t["page"], t["caption"]))
        print("    колонки: %s" % " | ".join(t["columns"]))
        for r in t["rows"]:
            print("    %-46s %s" % (r["label"][:46],
                  " ".join("%s=%s" % (t["columns"][c], v) for c, v in r["cells"])))
