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
    "column_header_after",
    "context_outside",
    "row_context",
    "values_in_row",
    "header_positions",
    "is_column_header",
    "table_context_before",
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
# ★ПОСЛЕДНЕЕ ЗНАЧЕНИЕ СТРОКИ ЗАКАНЧИВАЕТСЯ ПЕРЕВОДОМ СТРОКИ, А НЕ ПРОБЕЛОМ.
# Требование «за каждым значением идёт пробел или табуляция» отрезало у строки
# ровно один столбец — последний. Замерено на отчёте за июль 2026: у 903 строк
# из 938 сразу за концом фрагмента в отчёте стояло ещё одно число. Столбец этот
# не случайный: в таблицах STEO последняя колонка — годовая, то есть прогноз на
# 2027 год. Вопросы задают именно о нём.
_ROW_VALUES = re.compile(r"\s*(?:(?:-?[\d,]+(?:\.\d+)?|-)(?:[ \t]+|(?=\n|$))){4,}")

# Одиночное значение таблицы: число или прочерк на месте отсутствующих данных.
_VALUE_TOKEN = re.compile(r"^(?:-?[\d,]+(?:\.\d+)?|-)$")

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


# ── шапка колонок ──────────────────────────────────────────────────────────
# Заголовок называет таблицу, но не говорит, ЧТО ЗНАЧИТ каждое число в строке.
# Столбцы названы отдельной строкой сразу под заголовком:
#
#   Table 3c. World Petroleum and Other Liquid Fuels Production (million barrels per day)
#   Q1 Q2 Q3 Q4 Q1 Q2 Q3 Q4 Q1 Q2 Q3 Q4 2025 2026 2027
#   OPEC members subject to OPEC+ agreements ..... 21.55 21.96 22.38 …
#
# Без этой строки модель видит ряд чисел и не знает, какому периоду какое
# принадлежит. Это не догадка: на вопросе про добычу ОПЕК+ она так и ответила —
# «числа есть, заголовков колонок в фрагменте не видно» — и оговорила единицы
# как предположение. Строка короткая и дешёвая, а без неё табличный материал
# отвечает наполовину.
#
# ★РАСПОЗНАЁТСЯ ПО СОСТАВУ, А НЕ ПО ПОЛОЖЕНИЮ. «Строка сразу под заголовком»
# ошибочно: под ним бывает колонтитул «U.S. Energy Information Administration |
# Short-Term Energy Outlook - July 2026», и он содержит «July 2026». Строка
# признаётся шапкой, только если ВСЕ её слова — обозначения периодов.
COLUMN_LABEL = re.compile(
    r"^(?:Q[1-4]|Year|1[89]\d{2}|20\d{2}"
    r"|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$",
    re.IGNORECASE,
)

# Меньше трёх обозначений — не шапка: две подряд стоящие даты встречаются в
# прозе, а три и больше в одну строку ставит только таблица.
MIN_COLUMN_LABELS = 3

# Насколько далеко под заголовком искать первую строку таблицы. Между ними
# помещаются колонтитул, сноски и строка-раздел, но не половина отчёта.
HEADER_SEARCH_WINDOW = 6000

# Подряд идущие обозначения периодов. Ищем именно РЯД, а не строку целиком:
# извлекатель PDF регулярно приклеивает шапку к концу предыдущей фразы —
# «…Energy Information Administration.Q1 Q2 Q3 Q4 …», — и построчная проверка
# такую шапку не видит, хотя она есть.
_COLUMN_RUN = re.compile(
    r"\b(?:(?:Q[1-4]|Year|(?:19|20)\d{2}"
    r"|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(?:[ \t]+|(?=\n|$)))+",
    re.IGNORECASE,
)


def is_column_header(line: str) -> bool:
    """Все ли слова строки — обозначения периодов, и есть ли их достаточно."""
    tokens = line.split()
    if len(tokens) < MIN_COLUMN_LABELS:
        return False
    return all(COLUMN_LABEL.match(token) for token in tokens)


def values_in_row(row_text: str) -> int:
    """Сколько значений несёт строка таблицы: числа и прочерки после выноски."""
    dots = _DOTS.search(row_text)
    tail = row_text[dots.end():] if dots else row_text
    return sum(1 for token in tail.split() if _VALUE_TOKEN.match(token))


def column_header_after(stream: str, position: int) -> str:
    """Шапка колонок таблицы, заголовок которой начинается в ``position``.

    ★ШАПКА СТОИТ НЕПОСРЕДСТВЕННО ПЕРЕД ПЕРВОЙ СТРОКОЙ ТАБЛИЦЫ. Это не догадка о
    вёрстке, а замер: на отчёте за июль 2026 ряд «Q1 Q2 Q3 Q4 …» отстоит от
    первой строки на 39–51 знак во всех таблицах, где он вообще извлёкся.
    Поэтому область поиска — от заголовка до первой строки, а из найденных
    рядов берётся ПОСЛЕДНИЙ: выше него лежат сноски и годовые полосы.

    ★И ПРИНИМАЕТСЯ, ТОЛЬКО ЕСЛИ НАЗЫВАЕТ СТОЛЬКО СТОЛБЦОВ, СКОЛЬКО В СТРОКЕ
    ЗНАЧЕНИЙ. Без сверки правило ловит не то: 619 строк из 938 получали «шапку»
    из четырёх слов «2025 2026 2027 Year» при пятнадцати значениях в строке.
    Это не шапка, а обрывок годовой полосы, вынесенный извлекателем PDF
    отдельной строкой. Показать её модели значило бы привязать пятнадцать чисел
    к четырём периодам, и ответ выглядел бы обычным. Чужая шапка врёт
    незаметнее чужого заголовка.

    Пустая строка означает, что шапки нет, — законный исход: у части таблиц
    столбцы названы внутри самих строк, а из части их не извлёк конвертер.
    """
    window = stream[position:position + HEADER_SEARCH_WINDOW]
    spans = table_rows(window)
    if not spans:
        return ""
    first_row = spans[0][0]
    width = values_in_row(window[first_row:spans[0][1]])
    if width == 0:
        return ""

    for run in reversed(list(_COLUMN_RUN.finditer(window, 0, first_row))):
        candidate = run.group(0).strip()
        if is_column_header(candidate) and len(candidate.split()) == width:
            return candidate
    return ""


def header_positions(
    stream: str,
    captions: list[tuple[int, str]],
    known: dict[str, list[str]] | None = None,
) -> list[str]:
    """Шапки колонок, по одной на заголовок, в том же порядке.

    Как и заголовки, считаются один раз на документ: иначе окно в 400 знаков
    перечитывалось бы для каждого из тысяч фрагментов.

    ★ГОТОВАЯ ШАПКА ПРЕДПОЧТИТЕЛЬНЕЕ ВОССТАНОВЛЕННОЙ. ``known`` — заголовок
    таблицы → метки колонок, разобранные pdf2xml ПО КООРДИНАТАМ. Они несут
    привязку квартала к году (``2025Q1``), которой в плоском тексте нет и
    взяться ей неоткуда: верхний ярус шапки — отдельная строка потока, и
    сопоставить её с нижним можно только по x, то есть на этапе разбора.

    Разбор регулярками остаётся запасным путём — для таблиц, которых pdf2xml не
    собрал, и для вызовов, куда структуру не передали (её нет у синтетических
    страниц в тестах). Он честно отдаёт нижний ярус: хуже полной шапки, лучше
    пустой.
    """
    known = known or {}
    return [
        " ".join(known[caption]) if caption in known else column_header_after(stream, start)
        for start, caption in captions
    ]


def _nearest_caption(captions: list[tuple[int, str]], position: int) -> int:
    """Индекс ближайшего заголовка не позже ``position``; -1, если такого нет.

    Вынесено, чтобы у заголовка и у шапки был ОДИН ответ на вопрос «к какой
    таблице относится это место». Разойдись они — фрагмент получил бы заголовок
    одной таблицы и шапку другой, и заметить это было бы нечем.
    """
    from bisect import bisect_right

    return bisect_right([start for start, _ in captions], position) - 1


def table_context_before(
    captions: list[tuple[int, str]], headers: list[str], position: int
) -> str:
    """Заголовок таблицы и шапка её колонок для места ``position``.

    Обе части подчиняются одному правилу удалённости: чужая шапка так же врёт,
    как чужой заголовок, только незаметнее — числа привяжутся к периодам другой
    таблицы, и ошибка будет выглядеть как обычный ответ.
    """
    index = _nearest_caption(captions, position)
    if index < 0:
        return ""
    start, caption = captions[index]
    if position - start > MAX_CAPTION_DISTANCE:
        return ""
    header = headers[index] if index < len(headers) else ""
    return f"{caption}\n{header}" if header else caption


def context_outside(body: str, context: str) -> str:
    """Часть контекста, которой нет в самом фрагменте.

    Повторять то, что уже лежит в тексте, незачем: удвоенные слова размывают
    эмбеддинг. Части взвешиваются по отдельности, потому что заголовок и шапка
    попадают в тело фрагмента независимо друг от друга.
    """
    kept = [part for part in context.split("\n") if part and part not in body]
    return "\n".join(kept)


def row_context(row_text: str, context: str) -> str:
    """Контекст для ОДНОЙ строки таблицы, с поверкой шапки по этой же строке.

    Ширина сверялась при поиске шапки, но по ПЕРВОЙ строке таблицы. Часть строк
    несёт меньше значений — раздел без данных, показатель, которого нет в
    квартальном разрезе. Такой строке шапка не подходит, и подставлять её
    нельзя: пятнадцать периодов над шестью числами читаются как пропуск данных
    в конце, а на деле смещены все.

    Заголовок таблицы остаётся: он к ширине отношения не имеет.
    """
    caption, _, header = context.partition("\n")
    if header and len(header.split()) != values_in_row(row_text):
        context = caption
    return context_outside(row_text, context)


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
    index = _nearest_caption(captions, position)
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

    # Заголовок таблицы в потоке и в структуре — один и тот же текст, поэтому
    # он и служит ключом. Таблица, разбитая на две страницы, даёт одинаковые
    # метки с обеих: перезапись безвредна.
    known_columns: dict[str, list[str]] = {}
    for page in pages:
        for table in page.get("tables") or ():
            if table.get("caption") and table.get("columns"):
                known_columns[table["caption"]] = table["columns"]

    captions = caption_positions(stream)
    headers = header_positions(stream, captions, known_columns)

    step = size - overlap
    chunks: list[dict] = []
    start = 0
    total = len(stream)
    caption_starts = [position for position, _ in captions]
    while start < total:
        end = min(start + size, total)

        # ★ФРАГМЕНТ НЕ ИМЕЕТ ПРАВА ПЕРЕСЕКАТЬ ЗАГОЛОВОК ТАБЛИЦЫ.
        #
        # Окно шириной 1200 знаков регулярно начинается в сносках одной таблицы
        # и продолжается числами следующей. Подпись фрагменту выбирается по его
        # НАЧАЛУ — и приезжает чужая: контекст говорит «Table 3a … Consumption»,
        # а цифры внутри от «Table 3b … Production». Замер по корпусу: таких
        # фрагментов 225 из 9456, у 173 подпись расходится с той, что начинается
        # внутри. Это 1.8% корпуса, и бьёт оно ровно по табличным вопросам.
        #
        # ★ЦЕНА ОШИБКИ ЗДЕСЬ ВЫШЕ, ЧЕМ КАЖЕТСЯ. Модель читает цифры верно и
        # приписывает их таблице, названной в контексте. Получается ссылка,
        # которая выглядит проверяемой и ведёт не туда, — хуже, чем отсутствие
        # ответа: пустоту читатель заметит, подмену нет.
        #
        # Заголовок таблицы — граница смысла, а не случайное место в потоке.
        # Поэтому окно обрывается на нём, и следующее начинается ровно с него,
        # БЕЗ перекрытия: перекрытие втащило бы хвост предыдущей таблицы обратно
        # и вернуло бы ту же путаницу с другой стороны.
        cut = next((position for position in caption_starts if start < position < end), None)
        if cut is not None:
            end = cut

        body = stream[start:end]
        # Заголовок и шапка нужны только тем, кто их не содержит: продолжениям
        # таблицы. Если они уже внутри фрагмента, повтор ничего не добавит
        # эмбеддингу и лишь размоет его удвоенными словами.
        chunks.append(
            Chunk(
                index=len(chunks),
                text=body,
                start=start,
                end=end,
                page_start=page_of_char[start],
                page_end=page_of_char[end - 1],
                context=context_outside(body, table_context_before(captions, headers, start)),
            ).as_dict()
        )
        if end == total:
            break
        start = end if cut is not None else start + step

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
                context=row_context(
                    stream[row_start:row_end],
                    table_context_before(captions, headers, row_start),
                ),
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
