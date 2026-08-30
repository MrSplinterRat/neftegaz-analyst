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

import bisect
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
    "row_pairs",
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
#
# ★УТОЧНЕНИЕ 30.08.2026: ЦИФРА ВНУТРИ СЛОВА — ЧАСТЬ НАЗВАНИЯ, А НЕ ДАННЫЕ.
#
# Запрет цифр целиком стоил одного заголовка из двадцати шести: «Table 9a. U.S.
# Macroeconomic Indicators and CO2 Emissions» обрывался на «and CO», и обрывался
# ВСЕГДА — все 312 фрагментов этой таблицы в индексе несли подпись без формулы.
# Цена: фрагмент про выбросы не содержит слова CO2 ни в тексте подписи, ни в
# эмбеддинге, а отвечающая модель читает бессмысленное «and CO».
#
# Различает эти два случая не наличие цифры, а ЧТО СТОИТ ПЕРЕД НЕЙ. В названии
# цифра приклеена к букве (CO2, NO2, CH4); данные таблицы отделены пробелом
# («Energy Prices 2025Q1»). Поэтому цифра разрешена только сразу после буквы —
# ограничитель остаётся тем же по духу, но проходит по настоящей границе.
#
# Замер до правки: 26 различных подписей, 196 вхождений, цифр после номера
# таблицы НИ В ОДНОЙ ⇒ послабление не может задеть ничего, кроме исправляемого.
TABLE_CAPTION = re.compile(
    r"Table\s+\d+[a-z]?\.\s+[A-Z](?:[A-Za-z .,/&'-]|(?<=[A-Za-z])\d){4,80}(?:\([^)]{0,40}\))?"
)

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
# ★ЧИСЛО ОБЯЗАНО СОДЕРЖАТЬ ЦИФРУ. Прежняя форма `-?[\d,]+(?:\.\d+)?` требовала
# лишь знак из класса «цифра ИЛИ запятая» — то есть ОДИНОКАЯ ЗАПЯТАЯ считалась
# числом. Разделитель тысяч втащил в класс знак препинания, и тот стал сходить
# за значение везде, где `_NUMBER` применяется.
#
# Видно это было не там, где причина. В возврате подписи стоит охрана «три числа
# подряд в куске = это чужие значения, дальше не идти». На законном куске
# «(Index, 2017=100» она насчитывала [',', '2017', '100'] — три вместо двух — и
# отвергала возврат. Подписью оставалась закрывающая скобка, а фрагмент выходил
# немым: числа есть, сказать чего они нечем.
#
#     GDP Implicit Price Deflator
#     (Index, 2017=100) ............ 127.6 128.3 129.5 …
#
# Замер на отчёте за июль 2026: 6 строк из 938 — все с единицами измерения в
# скобках на отдельной строке, включая предельный случай Table 9a.
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
# ★ПОСЛЕДНЕЕ ЗНАЧЕНИЕ СТРОКИ ЗАКАНЧИВАЕТСЯ ПЕРЕВОДОМ СТРОКИ, А НЕ ПРОБЕЛОМ.
# Требование «за каждым значением идёт пробел или табуляция» отрезало у строки
# ровно один столбец — последний. Замерено на отчёте за июль 2026: у 903 строк
# из 938 сразу за концом фрагмента в отчёте стояло ещё одно число. Столбец этот
# не случайный: в таблицах STEO последняя колонка — годовая, то есть прогноз на
# 2027 год. Вопросы задают именно о нём.
_ROW_VALUES = re.compile(r"\s*(?:(?:-?[\d,]+(?:\.\d+)?|-)(?:[ \t]+|(?=\n|$))){4,}")

# Одиночное значение таблицы: число или прочерк на месте отсутствующих данных.
_VALUE_TOKEN = re.compile(r"^(?:-?[\d,]+(?:\.\d+)?|-)$")

# Прочерк, стоящий отдельным значением: за ним пробельное, а не буква. Отличает
# «- - Congo» (хвост чужой строки) от «millions - SAAR» (тире внутри подписи —
# там за дефисом идёт пробел, но перед ним стоит слово, поэтому срез идёт только
# от переднего края подписи, а не по всякому совпадению).
_DASH_VALUE = re.compile(r"-(?=\s|$)")

# Буквы в куске текста: признак того, что граница подписи отрезала имя, а не
# числа. Двух подряд довольно — единичная буква бывает пометкой сноски «(a)».
_LETTERS = re.compile(r"[A-Za-z]{2,}")

# Насколько далеко назад от точек искать начало подписи.
ROW_LABEL_WINDOW = 100


def table_rows(stream: str, barriers: list[tuple[int, str]] | None = None) -> list[tuple[int, int]]:
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

    ★ЗАГОЛОВОК ТАБЛИЦЫ И ЕЁ РАЗДЕЛ — ПРЕГРАДЫ, КОТОРЫЕ ГРАНИЦА НЕ ПЕРЕХОДИТ.
    Отсчёт от последнего числа молча уезжает в предыдущую строку, когда чисел в
    окне нет, — и это бывает верно, и бывает неверно, а по виду не отличается.
    Замер на отчёте за июль 2026: у 201 строки из 938 подпись переходит через
    перевод строки; 127 переходов законны (имя показателя, у которого единицы и
    числа стоят на следующей строке: «Dry Natural Gas Production» + «(billion
    cubic feet per day) …»), а 74 съедают название раздела — «Energy Production
    Crude Oil Production (a) …» вместо «Crude Oil Production (a) …».

    ★Съеденный раздел портит вдвойне. Подпись строки перестаёт быть подписью
    строки, и та же самая тема приезжает ВТОРОЙ раз — раздел уже стоит в
    контексте фрагмента отдельным полем. Различить законный переход от
    незаконного изнутри текста нельзя: обе предыдущие строки выглядят как
    короткая строка без чисел. Различает знание СНАРУЖИ — какие места потока
    являются заголовками и разделами; оно уже добыто разбором геометрии, и
    ``barriers`` их сюда приносит.

    Возвращаются именно ГРАНИЦЫ, а не текст: фрагмент обязан быть дословной
    подстрокой потока, иначе ссылка на страницу перестанет быть проверяемой.
    """
    stops = sorted(barriers or ())
    starts = [position for position, _ in stops]
    spans: list[tuple[int, int]] = []
    for dots in _DOTS.finditer(stream):
        window_start = max(0, dots.start() - ROW_LABEL_WINDOW)
        window = stream[window_start : dots.start()]
        numbers = list(_NUMBER.finditer(window))
        label_start = window_start + (numbers[-1].end() if numbers else 0)
        # Преграда, попавшая между началом подписи и точками, сдвигает начало за
        # свой конец: подпись строки не вправе включать чужой заголовок.
        barrier_edge = 0
        place = bisect.bisect_right(starts, label_start)
        # ★Преграда бывает НАЧАТА ЛЕВЕЕ границы и накрывать её собой: последнее
        # число окна стоит внутри самого названия раздела («Industrial Production
        # Indices (Index, 2017=100)» → граница садится на «)»). Такую преграду
        # поиск по началу не находит, и раздел въезжал в подпись обрубком — 15
        # строк из 938 на отчёте за июль 2026, включая предельный случай Table 9a.
        if place and stops[place - 1][0] + len(stops[place - 1][1]) > label_start:
            place -= 1
        while place < len(stops) and starts[place] < dots.start():
            barrier_edge = max(barrier_edge, stops[place][0] + len(stops[place][1]))
            place += 1
        label_start = max(label_start, barrier_edge)
        while label_start < dots.start() and stream[label_start] in " \t\n":
            label_start += 1
        # ★ПРОЧЕРК — ЗНАЧЕНИЕ, ХОТЯ И НЕ ЧИСЛО. На месте отсутствующих данных в
        # этих таблицах стоит «-», и рядом значений он значением ПРИЗНАЁТСЯ
        # (_ROW_VALUES), а при отсчёте границы подписи — нет: _NUMBER требует
        # цифру. Поэтому хвост «- - - - -» предыдущей строки уезжал в подпись
        # следующей: «- - Congo (Brazzaville) …» вместо «Congo (Brazzaville) …»,
        # 96 строк из 938 на отчёте за июль 2026 — каждая десятая.
        #
        # ★Срезается ТОЛЬКО ПЕРЕДНИЙ край, а не всякий одиночный дефис. Считать
        # дефис значением наравне с числом нельзя: в подписи «Housing Starts
        # (millions - SAAR)» он стои́т внутри имени, и граница уехала бы на
        # «SAAR)». Починка, ломающая соседнее, дороже дефекта.
        # Пробельным считается и перевод строки: последний прочерк строки стоит
        # перед ним, и проверка на «дефис-пробел» буквально его не находила.
        while _DASH_VALUE.match(stream, label_start, dots.start()):
            label_start += 1
            while label_start < dots.start() and stream[label_start] in " \t\n":
                label_start += 1
        # ★ЧИСЛО БЫВАЕТ ВНУТРИ САМОЙ ПОДПИСИ, и тогда граница режет имя.
        #
        # «Lower 48 States (excl GOA)» превращалось в «States (excl GOA)»,
        # «CAISO SP15 zone» — в «zone», «Real Gross Domestic Product (billion
        # chained 2017 dollars - SAAR)» — в «dollars - SAAR)». Замер на отчёте за
        # июль 2026: 38 строк из 938. Обрубок хуже, чем кажется: «States (excl
        # GOA)» без «Lower 48» не находится ни одним запросом про Нижние 48 и при
        # этом выглядит целой подписью.
        #
        # ★РАЗЛИЧАЕТ ВЫНОСКА, А НЕ ВИД ЧИСЛА. Законная граница — это конец
        # значений ПРЕДЫДУЩЕЙ строки, а у всякой строки значения отделены
        # точками-выноской. Значит, кусок слева от границы можно вернуть в
        # подпись, если в нём есть буквы и НЕТ выноски: буквы говорят, что
        # отрезано имя, а отсутствие выноски — что это имя не чужой строки.
        # Нижняя граница возврата — преграда, конец предыдущей строки и край
        # окна: дальше них подпись не принадлежит этой строке ни при каких
        # условиях.
        # ★ВОЗВРАТ НЕ ВПРАВЕ МЕНЯТЬ РЕШЕНИЕ «строка это или нет». Признание
        # строкой стоит ниже и опирается на то, что между числами и точками есть
        # подпись. Разрешив возврату дотягивать подпись, я превратил 14 не-строк
        # в строки — счётчик вырос с 938 до 952, и все четырнадцать оказались
        # обрубками слов вида «Refinery and blender net pr | oduction».
        if label_start >= dots.start():
            continue  # подписи между числами и точками нет — это не строка таблицы
        # Пол возврата: преграда, конец предыдущей строки и край окна. Ближайшая
        # преграда СЛЕВА от границы тоже считается — прямой просмотр её не видит,
        # он идёт от границы вправо, и без этого раздел снова въезжал в подпись
        # (4 случая из 938).
        # ★ОКНО ПОИСКА — ОГРАНИЧИТЕЛЬ СКОРОСТИ, А НЕ СМЫСЛА, и полом ему быть
        # нельзя: на странице 39 оно рубило посреди слова — «Refinery and blender
        # net pr | oduction», — потому что предыдущая строка длинная и её
        # значения кончаются дальше ста знаков. Пол возврата смысловой (преграда
        # и конец предыдущей строки), а от разрастания подписи стои́т отдельный
        # потолок вдвое шире окна.
        floor = max(barrier_edge, spans[-1][1] if spans else 0, dots.start() - 2 * ROW_LABEL_WINDOW)
        behind = bisect.bisect_right(starts, label_start) - 1
        if behind >= 0:
            floor = max(floor, stops[behind][0] + len(stops[behind][1]))
        while label_start > floor:
            line_start = stream.rfind("\n", 0, label_start) + 1
            if line_start == label_start and label_start:
                line_start = stream.rfind("\n", 0, label_start - 1) + 1
            piece_start = max(line_start, floor)
            piece = stream[piece_start:label_start]
            # ★Ряд чисел в куске — это чужие значения, а не часть имени. Выноски
            # у них может и не быть: на странице 39 предыдущая строка не
            # распозналась строкой вовсе, и возврат перешагнул её значения,
            # собрав подпись в 200 знаков — «on 7.41 8.21 … Natural gas».
            if not _LETTERS.search(piece) or _DOTS.search(piece):
                break
            # ★СЧИТАЮТСЯ ЯЧЕЙКИ, А НЕ ВХОЖДЕНИЯ ЦИФР, и это не придирка к слову.
            # Охрана называла себя «ряд чисел = чужие значения», а считала любую
            # цифру — в том числе слипшиеся внутри ОДНОГО слова: у куска
            # «(index, 1982-1984=1.00» их три, и законный возврат отвергался, а
            # подписью строки оставалась закрывающая скобка.
            #
            # ★ЧУЖАЯ ЯЧЕЙКА БЫВАЕТ ТРЁХ ВИДОВ: число, прочерк и ОБОЗНАЧЕНИЕ
            # ПЕРИОДА. Последнее — шапка колонок, и она такая же не-часть имени,
            # как ряд значений. Прежняя охрана останавливалась на ней по
            # случайности: из «Q1 Q2 Q3 2025» регулярка вылавливала «1», «2»,
            # «3» — то есть была права по итогу и неправа по причине. Убрав одну
            # ошибку, я обнажил вторую (тест про склеенную шапку это и поймал).
            # Отличает чужую ячейку то, что она стои́т ОТДЕЛЬНЫМ токеном; оба
            # признака — `_VALUE_TOKEN` и `COLUMN_LABEL` — в этом файле уже есть.
            foreign = sum(
                1
                for token in piece.split()
                if _VALUE_TOKEN.match(token) or COLUMN_LABEL.match(token)
            )
            if foreign >= 3:
                break
            label_start = piece_start
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
    tail = row_text[dots.end() :] if dots else row_text
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
    window = stream[position : position + HEADER_SEARCH_WINDOW]
    spans = table_rows(window)
    if not spans:
        return ""
    first_row = spans[0][0]
    width = values_in_row(window[first_row : spans[0][1]])
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
    captions: list[tuple[int, str]],
    headers: list[str],
    position: int,
    blocks: list[tuple[int, str]] | None = None,
) -> str:
    """Заголовок таблицы, её раздел и шапка колонок для места ``position``.

    Обе части подчиняются одному правилу удалённости: чужая шапка так же врёт,
    как чужой заголовок, только незаметнее — числа привяжутся к периодам другой
    таблицы, и ошибка будет выглядеть как обычный ответ.

    ★РАЗДЕЛ ПРИСТАВЛЯЕТСЯ К ЗАГОЛОВКУ, А НЕ ОТДЕЛЬНОЙ СТРОКОЙ. Контекст читается
    потребителями как «заголовок, перевод строки, шапка» (см. ``row_context``),
    и третья строка сломала бы этот разбор. «Table 9a … — Carbon Dioxide (CO2)
    Emissions» и по смыслу вернее: это уточнённое название того же места.

    ★Раздел берётся только ВНУТРИ своей таблицы — не раньше её заголовка.
    Без этого условия раздел предыдущей таблицы протёк бы в следующую, и
    протёк бы незаметно: он выглядит как настоящий и на вид проверяем.
    """
    index = _nearest_caption(captions, position)
    if index < 0:
        return ""
    start, caption = captions[index]
    if position - start > MAX_CAPTION_DISTANCE:
        return ""
    if blocks:
        place = _nearest_caption(blocks, position)
        if place >= 0 and blocks[place][0] >= start:
            caption = f"{caption} — {blocks[place][1]}"
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


def row_pairs(row_text: str, header: str) -> str:
    """Строка таблицы, разобранная в явные пары «столбец = значение».

    ★ЗАЧЕМ. Без этого модель получает два независимых списка — ряд названий
    столбцов отдельно, ряд чисел отдельно — и обязана сама сообразить, что
    третье число относится к третьему столбцу. Это выравнивание по счёту, ровно
    то, что мы запретили себе при разборе страницы: связь должна быть
    ПРОЧИТАНА, а не выведена из порядка. Странно требовать этого от своего
    кода и оставлять модели.

    ★ЦЕНА ОШИБКИ ТА ЖЕ, ЧТО У СМЕЩЁННОЙ СТРОКИ: сдвиг на один столбец даёт
    настоящее число не за тот период. Ответ выглядит обычным, проверить его
    можно только вернувшись к самой странице — а он для того и нужен, чтобы
    туда не возвращаться.

    ★ПАРЫ СТРОЯТСЯ, ТОЛЬКО ЕСЛИ ЧИСЛО ЗНАЧЕНИЙ ТОЧНО РАВНО ЧИСЛУ СТОЛБЦОВ.
    Никаких «подгоним по левому краю» и «лишние отбросим»: неравные длины
    означают, что либо шапка чужая, либо строка неполная, и в обоих случаях
    любая пара будет выдумкой. Пустая строка на выходе — законный исход, при
    нём остаётся прежнее поведение: заголовок таблицы без шапки.

    Прочерк остаётся прочерком: «Q3 = -» значит «данных нет за третий квартал»,
    и это утверждение, а не пропуск. Молча выбросив прочерки, мы сдвинули бы
    все последующие значения — та же ошибка, только изнутри.
    """
    labels = header.split()
    if not labels:
        return ""

    dots = _DOTS.search(row_text)
    if dots is None:
        return ""
    name = " ".join(row_text[: dots.start()].split())
    values = [token for token in row_text[dots.end() :].split() if _VALUE_TOKEN.match(token)]
    if len(values) != len(labels):
        return ""

    # strict: длины сверены строкой выше; молчаливое усечение здесь означало бы
    # ровно тот сдвиг столбцов, ради которого всё это и написано.
    pairs = ", ".join(f"{label} = {value}" for label, value in zip(labels, values, strict=True))
    return f"{name}: {pairs}" if name else pairs


def row_context(row_text: str, context: str) -> str:
    """Контекст для ОДНОЙ строки таблицы, с поверкой шапки по этой же строке.

    Ширина сверялась при поиске шапки, но по ПЕРВОЙ строке таблицы. Часть строк
    несёт меньше значений — раздел без данных, показатель, которого нет в
    квартальном разрезе. Такой строке шапка не подходит, и подставлять её
    нельзя: пятнадцать периодов над шестью числами читаются как пропуск данных
    в конце, а на деле смещены все.

    Заголовок таблицы остаётся: он к ширине отношения не имеет.

    ★КОГДА ШАПКА ПОДОШЛА, В КОНТЕКСТ ИДЁТ НЕ ОНА, А РАЗБОР ПО ПАРАМ (см.
    ``row_pairs``): «Q1 = 20.31, Q2 = 20.51 …». Ряд названий столбцов при этом
    не теряется — он весь внутри пар, — зато исчезает необходимость сопоставлять
    два списка по счёту. Если пары построить не удалось, поведение прежнее.
    """
    caption, _, header = context.partition("\n")
    if header and len(header.split()) != values_in_row(row_text):
        return context_outside(row_text, caption)

    pairs = row_pairs(row_text, header) if header else ""
    if pairs:
        context = f"{caption}\n{pairs}" if caption else pairs
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
    # Разделы таблиц в координатах ПОТОКА: страницы приходят со смещениями
    # внутри себя, а сшиваются подряд, поэтому к каждому прибавляется длина уже
    # уложенного. Пересчёт здесь, а не у поставщика: смещение потока — свойство
    # сборки, и знать о нём разбору страницы незачем.
    block_positions: list[tuple[int, str]] = []
    offset = 0
    for page in pages:
        for block in page.get("blocks") or ():
            block_positions.append((offset + block["at"], block["title"]))
        offset += len(page["text"])
    block_positions.sort()

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
                context=context_outside(
                    body, table_context_before(captions, headers, start, block_positions)
                ),
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
    # Преграды для границы подписи: заголовки таблиц и их разделы. Заголовок в
    # подпись строки не съезжал ни разу (замер: 0 из 938), но и не должен —
    # условие стоит там же, где раздел, потому что различает их только вид
    # текста, а вид — не гарантия.
    for row_start, row_end in table_rows(stream, sorted(captions + block_positions)):
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
                    table_context_before(captions, headers, row_start, block_positions),
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
