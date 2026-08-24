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

import re
from dataclasses import asdict, dataclass

__all__ = [
    "Chunk",
    "TABLE_CAPTION",
    "chunk_pages",
    "chunk_document",
    "caption_before",
    "caption_positions",
    "table_rows",
]

# Заголовок таблицы в отчётах STEO: «Table 3d. World Crude Oil Production
# (million barrels per day)». Замерено на корпусе: таких заголовков 9 видов,
# и они единственное место, где числовая таблица названа словами.
#
# ★ГРАНИЦА ЗАДАНА ОТСУТСТВИЕМ ЦИФР, А НЕ ДЛИНОЙ. Замер на отчёте за декабрь
# 2025: вариант «до 90 знаков не-перевода-строки» дал 24 заголовка, и в один из
# них утекли цифры таблицы; вариант без цифр даёт те же 24 и ноль утечек.
# Выигрыш небольшой, и честнее назвать его небольшим: текст, извлечённый из
# этих PDF, переводы строк ИМЕЕТ (4074 штуки, строка около 59 знаков), так что
# ограничитель по [^\n] в основном срабатывал. Класс символов без цифр надёжнее
# лишь тем, что не зависит от того, поставил ли конвертер перевод строки именно
# здесь. Скобка с единицами захватывается намеренно и закрывает заголовок:
# «million barrels per day» отличает добычу от цены, а дальше идёт тело таблицы.
TABLE_CAPTION = re.compile(r"Table\s+\d+[a-z]?\.\s+[A-Z][A-Za-z .,/&'-]{4,80}(?:\([^)]{0,40}\))?")

# Насколько далеко назад разрешено искать заголовок. Таблица STEO занимает одну
# или две страницы, то есть несколько тысяч знаков; за этим пределом ближайший
# заголовок относится уже к другой таблице, и подставлять его — врать.
MAX_CAPTION_DISTANCE = 8000


@dataclass(frozen=True)
class Chunk:
    """One retrievable fragment, with its position in the source document."""

    index: int
    text: str
    start: int
    end: int
    page_start: int
    page_end: int
    # ★Заголовок таблицы, к которой относится фрагмент, если сам фрагмент его
    # не содержит. Хранится ОТДЕЛЬНО от text и никогда в него не вклеивается:
    # text обязан дословно совпадать с тем, что стоит на странице отчёта, иначе
    # цитата перестаёт быть проверяемой — а проверяемость и есть предмет
    # поставки. Заголовок участвует только в вычислении эмбеддинга.
    context: str = ""
    # "window" — обычный скользящий фрагмент; "row" — одна строка таблицы.
    # Различие хранится, потому что оно видно пользователю: строка таблицы
    # выглядит в интерфейсе иначе, чем абзац, и знать, что перед ним, полезно.
    kind: str = "window"

    def as_dict(self) -> dict:
        return asdict(self)


# ── строки таблиц ──────────────────────────────────────────────────────────
# Строка таблицы в этих отчётах выглядит так:
#   «United States ................................ 13.28 13.51 13.78 …»
# то есть подпись, точки-выноска и ряд значений. Точки — надёжный якорь: в
# сплошной прозе три точки подряд встречаются лишь многоточием, а за ним не
# идут пять чисел.
_DOTS = re.compile(r"\.{3,}")
_NUMBER = re.compile(r"-?[\d,]+(?:\.\d+)?")
_ROW_VALUES = re.compile(r"\s*(?:(?:-?[\d,]+(?:\.\d+)?|-)[ \t]+){4,}")

# Насколько далеко назад от точек искать начало подписи.
ROW_LABEL_WINDOW = 100


def table_rows(stream: str) -> list[tuple[int, int]]:
    """Границы строк таблиц в потоке: список пар (начало подписи, конец чисел).

    ★ЗАЧЕМ ОТДЕЛЬНЫМИ ФРАГМЕНТАМИ. Замерено: строка «United States … 13.28 13.51
    13.78» лежала внутри чанка на 1200 знаков вместе с десятком других стран и
    сносками, и НИ ОДИН способ поиска её оттуда не доставал — ни векторный
    (эталон не входил даже в top-15), ни по словам (24-е место). Дело не в
    способе поиска, а в разбавлении: строка про США составляет пятую часть
    фрагмента, остальное — про других. Короткий фрагмент, целиком посвящённый
    одной стране, находится и тем, и другим.

    ★ГРАНИЦА ПОДПИСИ — ПОСЛЕДНЕЕ ЧИСЛО ПРЕДЫДУЩЕЙ СТРОКИ, а не «столько-то
    знаков назад» и не «ближайшая заглавная буква». Заглавная буква даёт
    обрубок: у строки «Natural gas (dollars per million Btu) …» ближайшая
    заглавная — это «B», и подписью становится «Btu)».

    Возвращаются именно ГРАНИЦЫ, а не текст: фрагмент обязан быть дословной
    подстрокой потока, иначе ссылка на страницу перестанет быть проверяемой.
    """
    spans: list[tuple[int, int]] = []
    for dots in _DOTS.finditer(stream):
        window_start = max(0, dots.start() - ROW_LABEL_WINDOW)
        window = stream[window_start:dots.start()]
        numbers = list(_NUMBER.finditer(window))
        label_start = window_start + (numbers[-1].end() if numbers else 0)
        while label_start < dots.start() and stream[label_start] in " \t\n":
            label_start += 1
        if label_start >= dots.start():
            continue  # подписи между числами и точками нет — это не строка таблицы
        values = _ROW_VALUES.match(stream, dots.end())
        if values is None:
            continue  # точки без ряда значений: обычное многоточие в прозе
        spans.append((label_start, values.end()))
    return spans


def caption_positions(stream: str) -> list[tuple[int, str]]:
    """Все заголовки таблиц в потоке с их позициями, по возрастанию.

    Считается ОДИН раз на документ. Искать заголовок заново для каждого чанка
    значило бы перечитывать весь поток тысячу раз — при 1962 чанках корпуса это
    квадратичная работа там, где хватает одного прохода и двоичного поиска.
    """
    return [(match.start(), match.group(0).strip()) for match in TABLE_CAPTION.finditer(stream)]


def caption_before(captions: list[tuple[int, str]], position: int) -> str:
    """Ближайший заголовок таблицы, начинающийся не позже ``position``.

    ★ЗАЧЕМ ЭТО ВООБЩЕ. Продолжение таблицы — это чанк вида «20.31 20.51 20.97
    20.49 …», в котором нет ни одного слова. Его эмбеддинг — шум: запрос «добыча
    нефти в США» не может к нему приблизиться, потому что сближать не с чем.
    Замерено на нашем корпусе: 180 фрагментов упоминают crude oil production, но
    заголовков таблиц нашлось лишь 9 видов — их несут только те чанки, что
    начинаются рядом с заголовком, а числовые продолжения остаются безымянными.
    Отсюда и брался ответ «данных по добыче в США нет» при том, что данные есть.

    Пустая строка возвращается в двух случаях: заголовка выше нет вовсе, и
    ★ближайший заголовок дальше ``MAX_CAPTION_DISTANCE``. Второй случай важен не
    меньше первого: чужой заголовок хуже отсутствующего — он не просто не
    поможет найти фрагмент, он привяжет числа к чужой таблице.
    """
    from bisect import bisect_right

    index = bisect_right([start for start, _ in captions], position) - 1
    if index < 0:
        return ""
    start, caption = captions[index]
    return caption if position - start <= MAX_CAPTION_DISTANCE else ""


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

    captions = caption_positions(stream)

    step = size - overlap
    chunks: list[dict] = []
    start = 0
    total = len(stream)
    while start < total:
        end = min(start + size, total)
        body = stream[start:end]
        # Заголовок нужен только тем, кто его не содержит: продолжениям таблицы.
        # Если он уже внутри фрагмента, повтор ничего не добавит эмбеддингу и
        # лишь размоет его удвоенными словами.
        caption = caption_before(captions, start)
        chunks.append(
            Chunk(
                index=len(chunks),
                text=body,
                start=start,
                end=end,
                page_start=page_of_char[start],
                page_end=page_of_char[end - 1],
                context="" if caption and caption in body else caption,
            ).as_dict()
        )
        if end == total:
            break
        start += step

    # ★Строки таблиц идут ДОПОЛНИТЕЛЬНЫМИ фрагментами, а не вместо оконных.
    # Окна нужны прозе: рассуждение о рынке не разложено по строкам с точками.
    # Строки нужны числам: только в отдельном коротком фрагменте значение по
    # одной стране перестаёт тонуть среди десятка других. Одно и то же место
    # отчёта оказывается в индексе дважды, в двух разрешениях — и это замысел,
    # а не дублирование по недосмотру.
    for row_start, row_end in table_rows(stream):
        chunks.append(
            Chunk(
                index=len(chunks),
                text=stream[row_start:row_end],
                start=row_start,
                end=row_end,
                page_start=page_of_char[row_start],
                page_end=page_of_char[row_end - 1],
                context=caption_before(captions, row_start),
                kind="row",
            ).as_dict()
        )
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
