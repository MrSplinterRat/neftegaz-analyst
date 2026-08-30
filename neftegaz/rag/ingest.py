"""Turning PDF reports into an indexed, citable corpus.

Metadata is derived from the filename, by convention:

    OPEC_MOMR_2025-03.pdf  ->  source_name "OPEC MOMR", date "март 2025"

A convention rather than a sidecar metadata file because the corpus is meant to
be extended by dropping a PDF into data/reports/ — any step a user can forget
is a step that will be forgotten, and a report indexed without a date produces
a citation that cannot be checked.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from neftegaz.config import settings
from neftegaz.rag.chunking import chunk_document
from neftegaz.rag.confidence import annotate_chunks
from neftegaz.rag.crosscheck import crosscheck_pdf
from neftegaz.rag.intake import OK, inspect_pdf

__all__ = [
    "DocumentMeta",
    "parse_filename",
    "read_pdf_pages",
    "mark_unreadable_pages",
    "ingest_file",
    "ingest_directory",
]

RU_MONTHS = {
    1: "январь",
    2: "февраль",
    3: "март",
    4: "апрель",
    5: "май",
    6: "июнь",
    7: "июль",
    8: "август",
    9: "сентябрь",
    10: "октябрь",
    11: "ноябрь",
    12: "декабрь",
}


@dataclass(frozen=True)
class DocumentMeta:
    source_name: str
    date: str


def parse_filename(path: str) -> DocumentMeta:
    """Derive a report name and date from the file name.

    Recognised shapes, in order: ``NAME_YYYY-MM``, ``NAME_YYYY``. Anything else
    keeps the stem as the name and leaves the date empty — the document is
    still indexed and still cited, just without a date, which is honest.
    """
    stem = os.path.splitext(os.path.basename(path))[0]

    found = re.search(r"[_-](\d{4})-(\d{2})$", stem)
    if found:
        year, month = int(found.group(1)), int(found.group(2))
        name = stem[: found.start()].replace("_", " ").strip()
        month_name = RU_MONTHS.get(month, str(month))
        return DocumentMeta(source_name=name, date=f"{month_name} {year}")

    found = re.search(r"[_-](\d{4})$", stem)
    if found:
        name = stem[: found.start()].replace("_", " ").strip()
        return DocumentMeta(source_name=name, date=found.group(1))

    return DocumentMeta(source_name=stem.replace("_", " ").strip(), date="")


def read_pdf_pages(path: str) -> list[dict]:
    """Extract text per page as ``{"page": 1-based int, "text": str}``.

    Pages that yield no text (scans, full-page charts) are kept as empty
    entries rather than dropped: dropping them would shift every subsequent
    page number, and a citation pointing at the wrong page is worse than no
    citation at all.

    ★ТЕКСТ СОБИРАЕТСЯ ИЗ КООРДИНАТ СЛОВ, А НЕ ИЗ ПОТОКА ИЗВЛЕКАТЕЛЯ.

    PDF — предпечатный макет: в файле лежат геометрические примитивы, и порядок
    их следования никак не обязан совпадать с порядком чтения. В корпусе EIA
    STEO заголовок таблицы приходит ПОСЛЕ первой строки её данных на 168
    страницах из 208 — на странице 32 отстоит от неё на 4483 знака. Нарезка,
    подписывающая строку ближайшим заголовком выше по потоку, брала предыдущую
    таблицу: замер по корпусу дал 79% таких строк, и увеличением окна поиска
    это не лечится — неверный заголовок стоит БЛИЖЕ верного.

    Библиотека pdf2xml сортирует слова по y и собирает строки заново, поэтому
    заголовок оказывается там, где он стоит на бумаге, — над своей таблицей.
    """
    from pdf2xml import parse_pdf

    document = parse_pdf(path)
    pages = []
    for page in document.pages:
        text = page.text()
        # Collapse the ragged whitespace PDF extraction produces; it wastes
        # context and hurts embedding quality without carrying meaning.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # ★СТРУКТУРА ТАБЛИЦ ЕДЕТ ДАЛЬШЕ ВМЕСТЕ С ТЕКСТОМ, А НЕ ВЫБРАСЫВАЕТСЯ.
        #
        # pdf2xml разбирает шапку по координатам и знает, какой год стоит над
        # каким кварталом: column_labels() отдаёт ['2025Q1', … '2027Q4', '2025',
        # '2026', '2027']. Если оставить одну page.text(), эта связь теряется —
        # в плоском потоке два яруса шапки становятся отдельными строками, и
        # нарезка, восстанавливающая шапку регулярками, берёт только нижний:
        # двенадцать безымянных Q1…Q4, по которым нельзя сказать, какого года
        # квартал. Из плоского текста связь года с кварталом не выводится в
        # принципе — её нужно не угадывать заново, а донести.
        tables = [
            {"caption": table.caption, "columns": table.column_labels()}
            for table in page.tables
            if table.caption
        ]
        pages.append({"page": page.number, "text": text, "tables": tables})
    return pages


def mark_unreadable_pages(path: str, pages: list[dict]) -> int:
    """Отметить страницы, которые НАШ разборщик не прочёл, хотя текст там есть.

    ★ЗАЧЕМ. «На странице ничего нет» и «страницу не удалось прочесть» дают
    одинаковый результат — пустую строку, — а означают противоположное. Первое
    сообщает, что данных нет; второе, что данные есть и мы их потеряли. Пока
    оба случая выглядят одинаково, второй неотличим от знания, и это худший
    сорт отказа: ответ «за апрель данных нет» звучит одинаково уверенно и когда
    он верен, и когда мы просто не смогли прочесть апрельскую страницу.

    ★ОТЛИЧИТЬ ИХ ИЗНУТРИ ОДНОГО ЧИТАТЕЛЯ НЕЛЬЗЯ. У разборщика, вернувшего ноль
    слов, нет способа узнать, пуста страница или он на ней сломался: и то и
    другое выглядит как отсутствие слов. Нужен свидетель СНАРУЖИ — второй
    читатель, устроенный иначе. Он здесь уже есть: poppler всё равно
    запускается для сверки путей, и его вывод по странице ничего не стоит
    дополнительно.

    Правило: наш текст пуст, а poppler на той же странице видит текст ⇒ страница
    помечается `unreadable=True`. Оба пусты ⇒ страница действительно пуста
    (`unreadable=False`). Poppler не ответил вовсе ⇒ пометки нет: недоказанное
    остаётся недоказанным, а не записывается в чистое.

    Возвращает число помеченных страниц. Список `pages` меняется на месте.
    """
    from neftegaz.rag.crosscheck import _read_poppler

    blank = [i for i, page in enumerate(pages) if not page["text"].strip()]
    if not blank:
        return 0

    witness = _read_poppler(path)
    if witness is None:
        # Свидетель не явился. Промолчать здесь правильнее, чем записать
        # страницы в пустые: иначе отсутствие проверки стало бы её успешным
        # прохождением.
        return 0

    marked = 0
    for i in blank:
        number = pages[i]["page"]
        # Страницы у poppler нумеруются с единицы и идут по порядку.
        seen = witness[number - 1] if 0 < number <= len(witness) else ""
        if seen.strip():
            pages[i]["unreadable"] = True
            pages[i]["unreadable_witness"] = "poppler"
            marked += 1
    return marked


def ingest_file(path: str, store=None, intake=None) -> int:
    """Index one PDF. Returns the number of chunks written.

    ``intake`` may be passed in when the caller has already run acceptance, so
    the file is not inspected twice; it is inspected here otherwise, because a
    chunk indexed without its caveat is a citation that looks verified.
    """
    from neftegaz.rag.store import get_store

    store = store or get_store()
    meta = parse_filename(path)
    pages = read_pdf_pages(path)
    mark_unreadable_pages(path, pages)

    if not any(page["text"].strip() for page in pages):
        return 0

    chunks = chunk_document(
        pages,
        source_name=meta.source_name,
        doc_date=meta.date,
        size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    # Drop chunks that are only whitespace: they embed to noise and can
    # out-rank real content on short queries.
    chunks = [c for c in chunks if c["text"].strip()]
    # ★ОГОВОРКА ПРИКЛЕИВАЕТСЯ ЗДЕСЬ, А НЕ ПРИ ОТВЕТЕ. К моменту ответа файла
    # уже нет под рукой — есть только фрагмент из индекса; если статус не
    # уехал вместе с ним, восстановить его будет неоткуда, и цитата со спорной
    # страницы станет неотличима от цитаты с чистой.
    annotate_chunks(chunks, crosscheck=crosscheck_pdf(path), intake=intake or inspect_pdf(path))
    return store.index(chunks)


def ingest_directory(directory: str | None = None, recreate: bool = False) -> dict:
    """Index every PDF in a directory. Returns a per-file report."""
    from neftegaz.rag.store import get_store

    directory = directory or settings.reports_dir
    store = get_store()
    if recreate:
        store.ensure_collection(recreate=True)

    results: dict[str, int] = {}
    if not os.path.isdir(directory):
        return results

    for name in sorted(os.listdir(directory)):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(directory, name)
        # ★ПРИЁМКА ГОВОРИТ, НО НЕ РЕШАЕТ. Файл с находками всё равно
        # индексируется: решение «брать или не брать» принимается снаружи, а
        # дело приёмки — чтобы оговорка не потерялась молча. Единственная
        # точка входа (inspect_pdf) позже заменится вызовом exlayout.
        report = inspect_pdf(path)
        if report.severity != OK:
            print(f"  приёмка: {report.summary()}")
        try:
            results[name] = ingest_file(path, store=store, intake=report)
        except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop the corpus
            results[name] = -1
            print(f"  ! {name}: {exc}")
    return results
