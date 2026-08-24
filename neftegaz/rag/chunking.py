"""Splitting report text into chunks that remember which page they came from.

The assignment requires every retrieved fragment to be attributable down to the
page. That constraint drives the design: pages are concatenated into one
character stream so that a chunk may legitimately span a page break, while a
parallel array records the source page of every character. A chunk then reports
the page it starts on and the page it ends on.

The alternative — chunking each page separately — never spans a break and so
never needs the bookkeeping, but it fragments sentences at every page boundary
and measurably hurts retrieval on reports whose paragraphs run across pages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

__all__ = ["Chunk", "chunk_pages", "chunk_document"]


@dataclass(frozen=True)
class Chunk:
    """One retrievable fragment, with its position in the source document."""

    index: int
    text: str
    start: int
    end: int
    page_start: int
    page_end: int

    def as_dict(self) -> dict:
        return asdict(self)


def chunk_pages(pages: list[dict], size: int, overlap: int) -> list[dict]:
    """Split a list of ``{"page": int, "text": str}`` into overlapping chunks.

    Pages are joined with no separator: any separator would be text that exists
    in no source page, and it would land inside chunks and inside the character
    offsets we report. Empty pages therefore contribute nothing but are still
    skipped over correctly, and page numbers need not be contiguous — scanned
    corpora routinely skip or repeat them.

    ``overlap`` characters are shared between neighbours so that a sentence cut
    by a boundary still appears whole in one of the two chunks.
    """
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    if overlap < 0:
        raise ValueError(f"overlap must not be negative, got {overlap}")
    if overlap >= size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than size ({size}), "
            "otherwise the window never advances"
        )

    parts: list[str] = []
    page_of_char: list[int] = []
    for page in pages:
        text = page["text"]
        parts.append(text)
        page_of_char.extend([page["page"]] * len(text))
    stream = "".join(parts)

    step = size - overlap
    chunks: list[dict] = []
    start = 0
    total = len(stream)
    while start < total:
        end = min(start + size, total)
        chunks.append(
            Chunk(
                index=len(chunks),
                text=stream[start:end],
                start=start,
                end=end,
                page_start=page_of_char[start],
                page_end=page_of_char[end - 1],
            ).as_dict()
        )
        if end == total:
            break
        start += step
    return chunks


def chunk_document(
    pages: list[dict],
    source_name: str,
    doc_date: str,
    size: int = 1200,
    overlap: int = 200,
) -> list[dict]:
    """Chunk one report and stamp each chunk with the metadata a citation needs.

    Defaults of 1200/200 characters are a compromise: large enough that a chunk
    usually carries a whole argument with its numbers, small enough that an
    embedding still represents one topic rather than averaging several.
    """
    enriched = []
    for chunk in chunk_pages(pages, size=size, overlap=overlap):
        chunk["source_name"] = source_name
        chunk["date"] = doc_date
        # The citation cites where the claim *starts*; page_end is kept so a
        # reader can tell that the fragment ran over a break.
        chunk["page"] = chunk["page_start"]
        enriched.append(chunk)
    return enriched
