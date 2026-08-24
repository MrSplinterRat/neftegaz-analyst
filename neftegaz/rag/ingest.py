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

__all__ = ["DocumentMeta", "parse_filename", "read_pdf_pages", "ingest_file", "ingest_directory"]

RU_MONTHS = {
    1: "январь", 2: "февраль", 3: "март", 4: "апрель", 5: "май", 6: "июнь",
    7: "июль", 8: "август", 9: "сентябрь", 10: "октябрь", 11: "ноябрь", 12: "декабрь",
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
    """
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - a malformed page must not stop the file
            text = ""
        # Collapse the ragged whitespace PDF extraction produces; it wastes
        # context and hurts embedding quality without carrying meaning.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        pages.append({"page": number, "text": text})
    return pages


def ingest_file(path: str, store=None) -> int:
    """Index one PDF. Returns the number of chunks written."""
    from neftegaz.rag.store import get_store

    store = store or get_store()
    meta = parse_filename(path)
    pages = read_pdf_pages(path)

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
        try:
            results[name] = ingest_file(path, store=store)
        except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop the corpus
            results[name] = -1
            print(f"  ! {name}: {exc}")
    return results
