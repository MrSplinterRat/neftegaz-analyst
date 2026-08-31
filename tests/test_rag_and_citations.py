"""Tests for chunking, citation formatting, routing and the web filter.

The citation tests are the load-bearing ones. A wrong number in a forecast is a
bad forecast; a wrong or invented citation makes an unverified claim look
verified, which is the failure this product exists to prevent.
"""

from __future__ import annotations

import pytest

from neftegaz.agent.graph import parse_horizon_days, parse_supply_change
from neftegaz.rag.chunking import chunk_document, chunk_pages, row_pairs
from neftegaz.rag.ingest import parse_filename
from neftegaz.tools.citations import format_answer, format_claim
from neftegaz.tools.web import DENY_DOMAINS, WebResult, _domain, _is_denied, _is_preferred

# ── chunking ───────────────────────────────────────────────────────────────


def test_chunks_cover_every_character():
    pages = [{"page": 1, "text": "a" * 100}, {"page": 2, "text": "b" * 100}]
    chunks = chunk_pages(pages, size=60, overlap=10)
    rebuilt = chunks[0]["text"]
    for chunk in chunks[1:]:
        rebuilt += chunk["text"][10:]  # drop the overlap
    assert rebuilt == "a" * 100 + "b" * 100


def test_chunk_spanning_a_page_break_reports_both_pages():
    pages = [{"page": 1, "text": "a" * 50}, {"page": 2, "text": "b" * 50}]
    chunks = chunk_pages(pages, size=80, overlap=0)
    assert chunks[0]["page_start"] == 1
    assert chunks[0]["page_end"] == 2


def test_empty_page_in_the_middle_does_not_shift_numbering():
    """A chart-only page yields no text but must not renumber what follows."""
    pages = [
        {"page": 5, "text": "x" * 30},
        {"page": 6, "text": ""},
        {"page": 8, "text": "y" * 30},
    ]
    chunks = chunk_pages(pages, size=30, overlap=0)
    assert chunks[0]["page_start"] == 5
    assert chunks[1]["page_start"] == 8


def test_overlap_must_be_smaller_than_size():
    """Otherwise the window never advances and the loop never terminates."""
    with pytest.raises(ValueError):
        chunk_pages([{"page": 1, "text": "abc"}], size=10, overlap=10)


def test_chunk_document_stamps_citation_metadata():
    chunks = chunk_document(
        [{"page": 3, "text": "z" * 40}],
        source_name="EIA STEO",
        doc_date="июль 2026",
        size=40,
        overlap=0,
    )
    assert chunks[0]["source_name"] == "EIA STEO"
    assert chunks[0]["date"] == "июль 2026"
    assert chunks[0]["page"] == 3


# ── заголовок таблицы в чанках-продолжениях ────────────────────────────────
# ★Добавлено 24.08.2026. Продолжение таблицы — фрагмент вида «20.31 20.51
# 20.97», в котором нет ни одного слова; его эмбеддинг это шум, и запрос про
# добычу нефти к нему не приблизится. Отсюда брался ответ «данных по добыче в
# США нет» при том, что в корпусе 180 фрагментов про crude oil production.

_CAPTION = "Table 3d. World Crude Oil Production (million barrels per day)"


def test_caption_is_bounded_by_digits_not_by_length():
    """★Заголовок обязан кончаться там, где начинается таблица.

    Замерено на отчёте за декабрь 2025: вариант «до 90 знаков
    не-перевода-строки» дал 24 заголовка, в один из них утекли цифры таблицы;
    вариант, ограниченный отсутствием цифр, даёт те же 24 и ноль утечек.
    Выигрыш небольшой — текст из этих PDF переводы строк имеет, — но граница по
    цифрам не зависит от того, поставил ли конвертер перевод строки именно тут.
    """
    from neftegaz.rag.chunking import TABLE_CAPTION

    found = TABLE_CAPTION.search(f"{_CAPTION} Q1 Q2 2024 13.21 13.45 United States")
    assert found is not None
    assert found.group(0).strip() == _CAPTION


def test_a_formula_inside_a_word_stays_in_the_caption():
    """★Цифра, приклеенная к букве, — часть названия, а не начало данных.

    «Table 9a. U.S. Macroeconomic Indicators and CO2 Emissions» обрывался на
    «and CO», и обрывался ВСЕГДА: все 312 фрагментов этой таблицы в индексе
    несли подпись без формулы, то есть фрагмент про выбросы не содержал слова
    CO2 ни в подписи, ни в эмбеддинге.
    """
    from neftegaz.rag.chunking import TABLE_CAPTION

    caption = "Table 9a. U.S. Macroeconomic Indicators and CO2 Emissions"
    found = TABLE_CAPTION.search(f"{caption}\nFood ..... 104.0 104.1")
    assert found is not None
    assert found.group(0).strip() == caption


def test_a_digit_after_a_space_still_ends_the_caption():
    """★Отрицательный контроль к предыдущему, и он несёт всю осторожность правки.

    Послабление обязано пройти по границе «цифра после БУКВЫ», а не «цифра
    вообще»: иначе шапка колонок, стоящая на той же строке, утечёт в заголовок и
    вернётся ровно та ошибка, ради которой цифры были запрещены.
    """
    from neftegaz.rag.chunking import TABLE_CAPTION

    found = TABLE_CAPTION.search("Table 2. Energy Prices 2025Q1 2025Q2 2025Q3 2.55 2.54")
    assert found is not None
    assert found.group(0).strip() == "Table 2. Energy Prices"


def test_continuation_chunk_receives_the_table_caption():
    stream = _CAPTION + " " + "United States .... 13.21 13.45 13.60 " * 30
    chunks = chunk_pages([{"page": 1, "text": stream}], size=300, overlap=50)

    tail = chunks[-1]
    assert _CAPTION not in tail["text"], "фрагмент-продолжение не должен содержать заголовок сам"
    assert tail["context"] == _CAPTION


def test_chunk_that_already_holds_the_caption_gets_no_duplicate():
    """Повтор не добавил бы смысла, а удвоенные слова размыли бы эмбеддинг."""
    stream = _CAPTION + " 13.21 13.45"
    chunks = chunk_pages([{"page": 1, "text": stream}], size=500, overlap=0)

    assert _CAPTION in chunks[0]["text"]
    assert chunks[0]["context"] == ""


def test_distant_caption_is_not_borrowed():
    """★Чужой заголовок ХУЖЕ отсутствующего: он привяжет числа к другой таблице.

    Между заголовком и фрагментом здесь больше MAX_CAPTION_DISTANCE знаков, то
    есть таблица давно кончилась и идёт другой материал.
    """
    from neftegaz.rag.chunking import MAX_CAPTION_DISTANCE

    stream = _CAPTION + " " + "x" * (MAX_CAPTION_DISTANCE + 2000)
    chunks = chunk_pages([{"page": 1, "text": stream}], size=400, overlap=0)

    assert chunks[-1]["context"] == ""


def test_embeddable_joins_caption_but_text_stays_verbatim():
    """★Заголовок идёт в вектор и НЕ идёт в text.

    text обязан дословно совпадать со страницей отчёта: на нём стоит ссылка,
    которую заказчик будет проверять по номеру страницы. Склей мы заголовок в
    text — фрагмент перестал бы соответствовать источнику, и проверяемость,
    ради которой всё это делается, исчезла бы незаметно.
    """
    from neftegaz.rag.store import embeddable

    chunk = {"text": "13.21 13.45 13.60", "context": _CAPTION}
    assert embeddable(chunk) == f"{_CAPTION}\n13.21 13.45 13.60"
    assert chunk["text"] == "13.21 13.45 13.60"
    assert embeddable({"text": "проза без таблицы", "context": ""}) == "проза без таблицы"


# ── строки таблиц отдельными фрагментами ───────────────────────────────────
# ★Замерено: строка «United States … 13.28 13.51» внутри чанка на 1200 знаков
# не доставалась НИКАКИМ поиском — ни векторным (не входила в top-15), ни по
# словам (24-е место). Причина не в способе поиска, а в разбавлении: строка про
# США составляет пятую часть фрагмента, остальное — про другие страны.

_ROW_STREAM = (
    "Table 3d. World Crude Oil Production (million barrels per day)\n"
    "Saudi Arabia ......... 9.01 9.12 9.30 9.25 9.40\n"
    "United States ........ 13.28 13.51 13.78 13.61 13.55\n"
    "Natural gas (dollars per million Btu) ..... 2.12 2.28 2.16 2.13 2.16\n"
)


def test_table_row_becomes_its_own_fragment():
    from neftegaz.rag.chunking import table_rows

    spans = table_rows(_ROW_STREAM)
    texts = [_ROW_STREAM[a:b] for a, b in spans]
    assert any(t.startswith("United States") for t in texts), texts


def test_row_label_is_whole_not_truncated_to_last_capital():
    """★Подпись отсчитывается от последнего числа, а не от заглавной буквы.

    От заглавной получался обрубок: у строки «Natural gas (dollars per million
    Btu) … 2.12» ближайшая заглавная — «B», и подписью становилось «Btu)».
    """
    from neftegaz.rag.chunking import table_rows

    texts = [_ROW_STREAM[a:b] for a, b in table_rows(_ROW_STREAM)]
    gas = [t for t in texts if "Btu" in t]
    assert gas, texts
    assert gas[0].startswith("Natural gas"), gas[0]


# ── граница подписи не переходит заголовок и раздел ────────────────────────
# Отсчёт от последнего числа уезжает в предыдущую строку, когда чисел в окне
# нет. Замер на отчёте за июль 2026: так делают 201 строка из 938, и 74 из них
# съедали название раздела — «Energy Production Crude Oil Production (a) …»
# вместо «Crude Oil Production (a) …».

_SECTION_STREAM = (
    "Industrial Production Indices (Index, 2017=100)\n"
    "Total Industrial Production ..... 102.2 102.3 107.1 104.8 105.9\n"
)


def _first_row(stream: str, barriers=None) -> str:
    from neftegaz.rag.chunking import table_rows

    spans = table_rows(stream, barriers)
    assert spans, stream
    return stream[spans[0][0] : spans[0][1]]


def test_the_label_does_not_swallow_the_section_above_it():
    """Раздел без единой цифры — самый частый случай: 74 строки из 938.

    Без преграды подпись начинается на разделе, потому что чисел в окне нет
    вовсе и граница уезжает до края окна.
    """
    stream = "Energy Production\nCrude Oil Production (a) ..... 13.28 13.51 13.78 13.61\n"
    assert _first_row(stream).startswith("Energy Production"), "проверка не различает"
    assert _first_row(stream, [(0, "Energy Production")]).startswith("Crude Oil Production")


def test_the_label_does_not_swallow_a_section_that_ends_in_digits():
    barriers = [(0, "Industrial Production Indices (Index, 2017=100)")]
    assert _first_row(_SECTION_STREAM, barriers).startswith("Total Industrial Production")


def test_a_barrier_that_covers_the_boundary_still_stops_it():
    """★Преграда бывает НАЧАТА ЛЕВЕЕ границы и накрывает её собой.

    Последнее число окна стоит внутри самого названия раздела — «(Index,
    2017=100)», — и граница садится на закрывающую скобку. Поиск преграды по
    её началу такую не находит: она начинается раньше границы. Замер: пятнадцать
    строк из 938 остались испорченными именно так, и среди них предельный случай
    корпуса — «) Emissions (million metric tons) Total Energy (d)» из Table 9a.
    """
    label = _first_row(_SECTION_STREAM, [(0, "Industrial Production Indices (Index, 2017=100)")])
    assert not label.startswith(")"), label


def test_the_label_does_not_start_with_the_previous_rows_dashes():
    """★Прочерк — значение, хотя и не число.

    На месте отсутствующих данных стои́т «-». Рядом значений он значением
    признаётся, а при отсчёте границы подписи — нет, и хвост «- - - - -»
    предыдущей строки уезжал в подпись следующей: «- - Congo (Brazzaville) …»
    вместо «Congo (Brazzaville) …». Замер: 96 строк из 938, каждая десятая.
    """
    from neftegaz.rag.chunking import table_rows

    stream = "Angola ..... 1.12 1.14 - - - -\nCongo (Brazzaville) ..... 0.26 0.27 0.25 0.24\n"
    congo = [stream[a:b] for a, b in table_rows(stream) if "Congo" in stream[a:b]]
    assert congo, stream
    assert congo[0].startswith("Congo"), congo[0]


def test_a_dash_inside_the_label_is_not_trimmed():
    """★Отрицательный контроль: срезается только ПЕРЕДНИЙ край.

    Считать всякий одиночный дефис значением нельзя — в подписи «Housing Starts
    (millions - SAAR)» он стои́т внутри имени, и граница уехала бы на «SAAR)».
    Починка, ломающая соседнее, дороже дефекта, который она чинит.
    """
    stream = "Housing Starts (millions - SAAR) ..... 1.36 1.34 1.38 1.40\n"
    assert _first_row(stream).startswith("Housing Starts")


# ── число ВНУТРИ подписи ───────────────────────────────────────────────────
# Граница отсчитывается от последнего числа, и когда число стоит в самом имени,
# имя режется: «Lower 48 States (excl GOA)» → «States (excl GOA)», «CAISO SP15
# zone» → «zone». Замер на отчёте за июль 2026: 38 строк из 938. Обрубок хуже,
# чем кажется: он не находится запросом и при этом выглядит целой подписью.

_INNER_NUMBER_STREAM = (
    "Alaska ..... 0.42 0.44 0.43 0.45\nLower 48 States (excl GOA) (c) ..... 11.2 11.4 11.5 11.6\n"
)


def test_a_number_inside_the_label_does_not_cut_it():
    assert _first_row(_INNER_NUMBER_STREAM).startswith("Alaska")
    rows = _INNER_NUMBER_STREAM
    from neftegaz.rag.chunking import table_rows

    lower = [rows[a:b] for a, b in table_rows(rows) if "Lower" in rows[a:b]]
    assert lower, rows
    assert lower[0].startswith("Lower 48 States"), lower[0]


def test_the_label_never_reaches_back_over_the_previous_rows_values():
    """★Отрицательный контроль возврата, и он несущий.

    Возврат отменяет ту самую границу, которая держит строки раздельно. Без
    условия «в куске нет ряда чисел» он перешагнул значения предыдущей строки
    там, где та вовсе не распозналась строкой, и собрал подпись в 200 знаков:
    «on 7.41 8.21 … Natural gas». Замер: подписей длиннее 120 знаков стало семь,
    сейчас ноль.
    """
    from neftegaz.rag.chunking import table_rows

    rows = _INNER_NUMBER_STREAM
    lower = [rows[a:b] for a, b in table_rows(rows) if "Lower" in rows[a:b]][0]
    assert "0.42" not in lower, lower


def test_a_section_is_not_pulled_into_the_label_by_the_return():
    """Возврат не вправе перешагнуть преграду — иначе он отменил бы починку,
    сделанную выше по этому же файлу."""
    stream = "Energy Production\nLower 48 States (excl GOA) ..... 11.2 11.4 11.5 11.6\n"
    assert _first_row(stream, [(0, "Energy Production")]).startswith("Lower 48 States")


# ── единицы измерения на отдельной строке ──────────────────────────────────
# Имя показателя стои́т одной строкой, единицы в скобках — следующей, и числа
# идут за ними. Возврат обязан склеить обе половины; когда он отказывал,
# подписью оставалась закрывающая скобка, и фрагмент выходил НЕМЫМ: числа есть,
# сказать, чего они, нечем. Замер на отчёте за июль 2026: 6 строк из 938.


def test_a_lone_comma_is_not_counted_as_a_number():
    """★Разделитель тысяч втащил запятую в класс «цифра или запятая».

    `_NUMBER` требовал лишь знак из этого класса, поэтому ОДИНОКАЯ ЗАПЯТАЯ
    считалась числом: у куска «(Index, 2017=100» их выходило три вместо двух, и
    охрана «ряд чисел = чужие значения» отвергала законный возврат.
    """
    from neftegaz.rag.chunking import _NUMBER

    assert _NUMBER.findall("(Index, 2017=100") == ["2017", "100"]

    stream = (
        "Percent change from prior year ..... 2.0 2.1 2.3 2.0\n"
        "GDP Implicit Price Deflator\n"
        "(Index, 2017=100) ..... 127.6 128.3 129.5 130.6\n"
    )
    from neftegaz.rag.chunking import table_rows

    deflator = [stream[a:b] for a, b in table_rows(stream) if "127.6" in stream[a:b]]
    assert deflator, stream
    assert deflator[0].startswith("GDP Implicit Price Deflator"), deflator[0]


def test_numbers_glued_inside_one_word_are_not_a_row_of_values():
    """★Охрана считает ЗНАЧЕНИЯ, а не вхождения цифр.

    В «(index, 1982-1984=1.00)» три настоящих числа, но все внутри одного слова.
    Чужое значение отличает то, что оно стои́т отдельным токеном. Предельный
    случай корпуса — Table 9a, с. 52.
    """
    from neftegaz.rag.chunking import table_rows

    stream = (
        "Natural Gas-weighted manufacturing (b) ..... 94.1 94.3 95.7 94.2\n"
        "Consumer Price Index (all urban consumers) (a)\n"
        "(index, 1982-1984=1.00) ..... 3.19 3.21 3.23 3.26\n"
    )
    consumer = [stream[a:b] for a, b in table_rows(stream) if "3.19" in stream[a:b]]
    assert consumer, stream
    assert consumer[0].startswith("Consumer Price Index"), consumer[0]


def test_three_separate_values_still_stop_the_return():
    """★Отрицательный контроль ослабленной охраны, и он несущий.

    Считать значения вместо цифр — значит пропускать больше. Пропустить ряд
    ОТДЕЛЬНЫХ значений нельзя: ровно на нём возврат однажды перешагнул чужую
    строку, не распознанную строкой, и собрал подпись в 200 знаков. Здесь у
    предыдущей строки нет выноски из точек, то есть строкой она не признана, —
    и остановить возврат обязаны именно её значения.
    """
    from neftegaz.rag.chunking import table_rows

    stream = "Crude oil 7.41 8.21 9.02\nNatural gas ..... 2.37 2.38 2.59 2.55\n"
    gas = [stream[a:b] for a, b in table_rows(stream) if "2.37" in stream[a:b]]
    assert gas, stream
    assert "7.41" not in gas[0], gas[0]
    assert gas[0].startswith("Natural gas"), gas[0]


def test_a_column_header_also_stops_the_return():
    """★Чужая ячейка бывает и обозначением периода, а не только числом.

    Шапка колонок — такая же не-часть имени строки, как ряд значений. Прежняя
    охрана останавливалась на ней ПО СЛУЧАЙНОСТИ: из «Q1 Q2 Q3 2025» регулярка
    чисел вылавливала «1», «2», «3» внутри самих обозначений — то есть была
    права по итогу и неправа по причине. Как только счёт стал вестись по
    отдельным ячейкам, случайная опора исчезла, и подпись строки уехала в
    предыдущее предложение вместе со всей шапкой.
    """
    from neftegaz.rag.chunking import table_rows

    stream = (
        "Weather forecasts from the agency.Q1 Q2 Q3 2025\n"
        "West Texas Intermediate Spot Average ..... 71.85 64.63 65.78 65.40\n"
    )
    wti = [stream[a:b] for a, b in table_rows(stream) if "71.85" in stream[a:b]]
    assert wti, stream
    assert wti[0].startswith("West Texas Intermediate"), wti[0]


# ── позиция раздела в тексте страницы ──────────────────────────────────────


def test_a_section_is_located_as_a_whole_line_not_as_a_substring():
    """★Название раздела встречается и внутри заголовка таблицы.

    «Macroeconomic» стои́т в «Table 9a. U.S. Macroeconomic Indicators and CO2
    Emissions», и поиск подстрокой ставил раздел туда — левее его настоящего
    места. Название при этом верное, неверна позиция; зонд, печатающий название,
    такого не показывает.
    """
    from neftegaz.rag.ingest import locate_blocks

    text = "Table 9a. U.S. Macroeconomic Indicators and CO2 Emissions\nMacroeconomic\nReal GDP ..... 23,548 23,771\n"
    assert locate_blocks(text, ["Macroeconomic"]) == [{"title": "Macroeconomic", "at": 58}]


def test_the_same_section_twice_on_a_page_is_found_twice():
    """Курсор движется: две таблицы на странице несут одноимённые разделы, и без
    движущегося курсора оба раза находилось бы первое вхождение."""
    from neftegaz.rag.ingest import locate_blocks

    text = "Production\nA ..... 1 2 3 4\nProduction\nB ..... 5 6 7 8\n"
    assert [block["at"] for block in locate_blocks(text, ["Production", "Production"])] == [0, 27]


def test_without_barriers_a_two_line_label_still_joins():
    """★Отрицательный контроль: переход через строку сам по себе ЗАКОНЕН.

    Имя показателя и его единицы стоят на разных строках — «Dry Natural Gas
    Production» и «(billion cubic feet per day) …», — и таких переходов 131 из
    201. Запрет переходить строку вообще был бы починкой, ломающей больше, чем
    чинит; преграда ставится по знанию о разделах, а не по виду текста.
    """
    stream = (
        "Dry Natural Gas Production\n(billion cubic feet per day) ..... 116.5 117.2 118.0 119.1\n"
    )
    assert _first_row(stream).startswith("Dry Natural Gas Production")


def test_row_fragments_are_verbatim_substrings():
    """★Дословность — условие проверяемости ссылки, а не аккуратность.

    Заказчик открывает отчёт на названной странице и ищет фрагмент глазами.
    Стоит фрагменту разойтись с источником хоть пробелом — проверка перестаёт
    сходиться, и доверие к остальным ссылкам падает вместе с этой.
    """
    chunks = chunk_pages([{"page": 7, "text": _ROW_STREAM}], size=200, overlap=40)
    for chunk in chunks:
        assert _ROW_STREAM[chunk["start"] : chunk["end"]] == chunk["text"]


def test_prose_ellipsis_is_not_mistaken_for_a_table_row():
    """Многоточие в прозе — не строка таблицы: за ним не идёт ряд чисел."""
    from neftegaz.rag.chunking import table_rows

    assert table_rows("Рынок замер... и никто не знал, что будет дальше.") == []


def test_row_carries_its_table_caption():
    from neftegaz.rag.chunking import chunk_pages as split

    rows = [
        c
        for c in split([{"page": 7, "text": _ROW_STREAM}], size=200, overlap=40)
        if c["kind"] == "row" and c["text"].startswith("United States")
    ]
    assert rows
    assert "World Crude Oil Production" in rows[0]["context"]


# ── раздел таблицы в контексте строки ──────────────────────────────────────
# ★Одна таблица бывает про разное. «Table 9a. U.S. Macroeconomic Indicators and
# CO2 Emissions» несёт пять разделов, и строка «Chemicals» относится к
# промышленному производству, а не к выбросам. Заголовок, подставленный всем
# строкам, раздаёт им все темы разом: замерено на корпусе STEO — слово CO2
# досталось всем 312 строкам таблицы, из которых к выбросам относятся четыре.

_BLOCK_STREAM = (
    "Table 9a. U.S. Macroeconomic Indicators and CO2 Emissions\n"
    "Industrial Production Indices (Index, 2017=100)\n"
    "Chemicals ............ 102.2 102.3 107.1 104.8 105.9\n"
    "Carbon Dioxide (CO2) Emissions (million metric tons)\n"
    "Petroleum ............ 2,203 2,187 2,201 2,178 2,190\n"
)


def _block_page(stream: str = _BLOCK_STREAM) -> dict:
    titles = (
        "Industrial Production Indices (Index, 2017=100)",
        "Carbon Dioxide (CO2) Emissions (million metric tons)",
    )
    return {
        "page": 52,
        "text": stream,
        "blocks": [{"title": title, "at": stream.find(title)} for title in titles],
    }


def _row_context(chunks: list[dict], label: str) -> str:
    """Контекст строки, найденной ПО ВХОЖДЕНИЮ подписи, а не по началу текста.

    ★Фрагмент строки начинается не с подписи, когда предыдущая строка кончается
    цифрой или скобкой: у «Chemicals» под разделом «(Index, 2017=100)» граница
    подписи отсчитывается от последнего числа и захватывает закрывающую скобку
    предыдущей строки. Это отдельный дефект границы, он виден и на корпусе
    («) Total Industrial Production»), и к разделам отношения не имеет —
    поэтому проверка разделов не должна на него опираться.
    """
    rows = [c for c in chunks if c["kind"] == "row" and label in c["text"]]
    assert rows, [c["text"][:30] for c in chunks]
    return rows[0]["context"]


def test_row_takes_the_section_it_stands_in_not_just_the_caption():
    chunks = chunk_pages([_block_page()], size=200, overlap=40)
    assert "Industrial Production" in _row_context(chunks, "Chemicals")
    assert "Carbon Dioxide" in _row_context(chunks, "Petroleum")


def test_the_section_of_a_neighbour_row_does_not_leak():
    """★Отрицательный контроль, без которого проверка выше ничего не значит.

    Разбор, приставляющий к строке ЛЮБОЙ раздел страницы, прошёл бы предыдущую
    проверку наполовину и выглядел бы работающим на счётчике «у скольких строк
    есть раздел». Различает только то, что чужой раздел ОТСУТСТВУЕТ.
    """
    chunks = chunk_pages([_block_page()], size=200, overlap=40)
    assert "Carbon Dioxide" not in _row_context(chunks, "Chemicals")
    assert "Industrial Production" not in _row_context(chunks, "Petroleum")


def test_a_section_standing_before_the_caption_is_not_borrowed():
    """Раздел предыдущей таблицы не протекает в следующую.

    Протёк бы незаметно: он выглядит как настоящий раздел и на вид проверяем —
    строка получила бы тему таблицы, которой на этом месте уже нет.

    ★У ВТОРОЙ ТАБЛИЦЫ РАЗДЕЛОВ НЕТ ВОВСЕ, и это условие проверки, а не деталь
    оформления. Первая версия этого теста ставила чужой раздел перед таблицей, у
    которой были свои: ближайшим сверху всё равно оказывался свой, охранное
    условие не участвовало, и проверка проходила бы и без него — то есть не
    проверяла ничего.

    ★ВЫЗОВ ПРЯМОЙ, А НЕ ЧЕРЕЗ ``chunk_pages``. Граница подписи строки отступает
    назад до последнего числа и на коротком потоке уезжает в самый заголовок
    («Table 8.»), так что позиция строки оказывается ВЫШЕ раздела и раздел не
    применяется по совсем другой причине. Проверка условия должна зависеть от
    условия, а не от чужого дефекта.
    """
    from neftegaz.rag.chunking import table_context_before

    captions = [(0, "Table 8. Refinery Utilization"), (100, "Table 9a. Macro and CO2")]
    blocks = [(40, "Gross Inputs")]
    # Положительная половина: своя таблица — раздел приезжает.
    assert "Gross Inputs" in table_context_before(captions, ["", ""], 60, blocks)
    # Отрицательная: строка следующей таблицы — тот же раздел ближайший сверху,
    # и отвергает его именно условие «не раньше своего заголовка».
    assert "Gross Inputs" not in table_context_before(captions, ["", ""], 140, blocks)


def test_a_table_without_sections_leaves_the_caption_alone():
    """★Пустой раздел — законный исход. Без этой проверки не отличить разбор,
    приставляющий раздел всегда, от разбора, приставляющего его по делу."""
    chunks = chunk_pages([{"page": 7, "text": _ROW_STREAM}], size=200, overlap=40)
    assert " — " not in _row_context(chunks, "United States")


def test_quarter_columns_are_tied_to_their_year():
    """★Шапка обязана говорить, КАКОГО ГОДА квартал.

    Без этого фрагмент несёт двенадцать безымянных Q1…Q4, и вопрос «что
    происходит с добычей ОПЕК+» неотвечаем: числа есть, привязать не к чему.
    Провал во II квартале 2026 неотличим от провала во II квартале 2025.

    Связь года с кварталом живёт в ВЕРХНЕМ ярусе шапки, а он в плоском потоке —
    отдельная строка. Сопоставить ярусы можно только по x, то есть при разборе;
    здесь проверяется, что разобранное дошло до фрагмента, а не было
    восстановлено регулярками заново.
    """
    from neftegaz.rag.chunking import chunk_pages as split

    caption = "Table 3d. World Crude Oil Production (million barrels per day)"
    page = {
        "page": 38,
        "text": _ROW_STREAM,
        # Меток ровно столько, сколько значений в строках потока: шапка иной
        # ширины отбрасывается намеренно, и тест проверял бы не то.
        "tables": [
            {
                "caption": caption,
                "columns": ["2025Q1", "2025Q2", "2025Q3", "2025Q4", "2025"],
            }
        ],
    }
    rows = [
        c
        for c in split([page], size=200, overlap=40)
        if c["kind"] == "row" and c["text"].startswith("United States")
    ]
    assert rows, "строка таблицы не выделилась во фрагмент"
    header = rows[0]["context"].split("\n")[-1]
    assert "2025Q2" in header, f"квартал не привязан к году: {header!r}"


def test_header_falls_back_when_structure_is_absent():
    """Без структуры путь не рушится, а отдаёт нижний ярус.

    Страницы без разобранных таблиц — не экзотика: часть таблиц pdf2xml не
    собирает, и синтетические страницы в тестах структуры не имеют вовсе.
    Пустая шапка хуже неполной, поэтому запасной разбор остаётся.
    """
    from neftegaz.rag.chunking import chunk_pages as split

    rows = [
        c
        for c in split([{"page": 7, "text": _ROW_STREAM}], size=200, overlap=40)
        if c["kind"] == "row" and c["text"].startswith("United States")
    ]
    assert rows
    assert "World Crude Oil Production" in rows[0]["context"]


# ── filename metadata ──────────────────────────────────────────────────────


def test_filename_with_year_and_month():
    meta = parse_filename("/data/reports/EIA_STEO_2026-07.pdf")
    assert meta.source_name == "EIA STEO"
    assert meta.date == "июль 2026"


def test_filename_with_year_only():
    meta = parse_filename("OPEC_Annual_2025.pdf")
    assert meta.source_name == "OPEC Annual"
    assert meta.date == "2025"


def test_filename_without_date_still_yields_a_name():
    """Indexed and cited without a date is honest; refusing to index is not."""
    meta = parse_filename("some_report.pdf")
    assert meta.source_name == "some report"
    assert meta.date == ""


# ── citations ──────────────────────────────────────────────────────────────


def test_report_citation_exact_format():
    claim = {
        "source_type": "report",
        "text": "Спрос вырастет на 1.2 млн барр./сут.",
        "source_name": "OPEC MOMR",
        "date": "март 2025",
        "page": 14,
        # Ступень достоверности — часть ссылки; у прочитанного напрямую метка
        # остаётся чистой, чтобы оговорка на спорном фрагменте была заметна.
        "confidence": "direct",
    }
    assert format_claim(claim) == (
        "Спрос вырастет на 1.2 млн барр./сут. [Отчёт OPEC MOMR, март 2025, с. 14]"
    )


def test_claim_without_confidence_says_the_check_did_not_run():
    """Missing status is «not checked», never «checked and clean»."""
    claim = {
        "source_type": "report",
        "text": "Спрос вырастет.",
        "source_name": "OPEC MOMR",
        "date": "март 2025",
        "page": 14,
    }
    assert format_claim(claim).endswith("с. 14; сверка чтения не выполнялась]")


def test_disputed_fragment_carries_the_warning_into_the_citation():
    """The whole point of Pareto-3: a contested page cannot look verified."""
    claim = {
        "source_type": "report",
        "text": "Добыча 13.28 млн барр./сут.",
        "source_name": "EIA STEO",
        "date": "июль 2026",
        "page": 22,
        "confidence": "disputed",
    }
    rendered = format_claim(claim)
    assert "с. 22; ⚠ два пути чтения расходятся по цифрам]" in rendered


def test_geometry_fragment_says_how_it_was_assembled():
    claim = {
        "source_type": "report",
        "text": "Строка таблицы.",
        "source_name": "EIA STEO",
        "date": "июль 2026",
        "page": 30,
        "confidence": "geometry",
    }
    assert format_claim(claim).endswith("с. 30; текст собран по геометрии страницы]")


def test_web_citation_exact_format():
    claim = {"source_type": "web", "text": "Brent торгуется у 92 долл.", "source_name": "Reuters"}
    assert format_claim(claim) == "Brent торгуется у 92 долл. [Источник: Reuters, web]"


def test_unknown_source_type_is_refused():
    with pytest.raises(ValueError):
        format_claim({"source_type": "guess", "text": "x", "source_name": "y"})


def test_report_claim_without_page_is_refused():
    """A page cannot be invented: it is what makes the citation checkable."""
    with pytest.raises(KeyError):
        format_claim(
            {"source_type": "report", "text": "x", "source_name": "OPEC", "date": "март 2025"}
        )


def test_mixed_answer_marks_each_paragraph_separately():
    answer = format_answer(
        [
            {
                "source_type": "report",
                "text": "Запасы снизились.",
                "source_name": "EIA STEO",
                "date": "июль 2026",
                "page": 7,
                "confidence": "direct",
            },
            {"source_type": "web", "text": "Цена выросла сегодня.", "source_name": "Reuters"},
        ]
    )
    assert "[Отчёт EIA STEO, июль 2026, с. 7]" in answer
    assert "[Источник: Reuters, web]" in answer
    assert answer.count("\n\n") == 1


# ── web filtering ──────────────────────────────────────────────────────────


def test_domain_extraction_strips_www():
    assert _domain("https://www.reuters.com/business/energy") == "reuters.com"


def test_tabloid_is_denied_including_subdomains():
    assert _is_denied("dailymail.co.uk")
    assert _is_denied("news.dailymail.co.uk")


def test_unknown_domain_is_allowed_but_not_preferred():
    assert not _is_denied("some-energy-blog.example")
    assert not _is_preferred("some-energy-blog.example")


def test_agency_domain_is_preferred():
    assert _is_preferred("eia.gov")
    assert _is_preferred("www.reuters.com".removeprefix("www."))


def test_deny_list_and_preferred_list_do_not_overlap():
    """A domain in both lists would make ranking depend on evaluation order."""
    from neftegaz.tools.web import PREFERRED_DOMAINS

    assert not (DENY_DOMAINS & PREFERRED_DOMAINS)


def test_web_result_becomes_a_web_claim():
    result = WebResult(
        title="Reuters",
        url="https://reuters.com/x",
        snippet="текст",
        domain="reuters.com",
        preferred=True,
    )
    claim = result.as_claim()
    assert claim["source_type"] == "web"
    assert format_claim(claim).endswith("[Источник: Reuters, web]")


# ── request parsing ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "question,expected",
    [
        ("спрогнозируй Brent на 3 месяца", 90),
        ("прогноз на 2 недели", 14),
        ("что будет через 1 год", 365),
        ("оцени на 45 дней", 45),
        ("спрогнозируй цену", 90),  # default
        ("прогноз на 5 лет", 730),  # clamped
    ],
)
def test_horizon_parsing(question, expected):
    assert parse_horizon_days(question) == expected


@pytest.mark.parametrize(
    "question,expected",
    [
        ("при сокращении добычи на 1.5 млн барр/сут", -1.5),
        ("если добыча вырастет на 2 млн баррелей в сутки", 2.0),
        ("что с ценой Brent", 0.0),
    ],
)
def test_supply_scenario_parsing(question, expected):
    assert parse_supply_change(question) == pytest.approx(expected)


# ── разбор ответа классификатора ───────────────────────────────────────────
# Эти тесты закрывают дефект, из-за которого КАЖДЫЙ запрос уходил в ветку
# forecast: разбор искал первое вхождение подстроки, а рассуждающая модель
# перечисляет все категории внутри <think> прежде, чем выбрать одну.


def test_route_plain_verdict():
    from neftegaz.agent.graph import parse_route

    assert parse_route("forecast") == "forecast"
    assert parse_route("  Industry  ") == "industry"
    assert parse_route("other") == "other"


def test_route_ignores_reasoning_block():
    """Вердикт стоит ПОСЛЕ рассуждения, а внутри упомянуты все варианты."""
    from neftegaz.agent.graph import parse_route

    verdict = (
        "<think>\n"
        "*   `forecast`: просят прогноз цены -> не подходит.\n"
        "*   `industry`: вопрос по отрасли -> не подходит.\n"
        "*   `other`: кулинария -> подходит.\n"
        "</think>\n"
        "other"
    )
    assert parse_route(verdict) == "other"


def test_route_survives_truncated_reasoning():
    """Обрыв по лимиту токенов оставляет <think> незакрытым."""
    from neftegaz.agent.graph import parse_route

    assert parse_route("<think>размышляю про forecast и industry") == "industry"


def test_route_takes_last_match_not_first():
    """Решение модели — то, что она сказала последним."""
    from neftegaz.agent.graph import parse_route

    assert parse_route("это не forecast и не industry, а other") == "other"


def test_route_matches_whole_words_only():
    from neftegaz.agent.graph import parse_route

    assert parse_route("forecasting is not the task") == "industry"


def test_route_defaults_to_industry_on_garbage():
    """Отказать по профильному вопросу хуже, чем ответить на непрофильный."""
    from neftegaz.agent.graph import parse_route

    assert parse_route("не понял вопроса") == "industry"
    assert parse_route("") == "industry"


def test_strip_reasoning_removes_block_and_keeps_answer():
    from neftegaz.agent.llm import strip_reasoning

    assert strip_reasoning("<think>долгие раздумья</think>\nОтвет") == "Ответ"
    assert strip_reasoning("Ответ без раздумий") == "Ответ без раздумий"


def test_strip_reasoning_drops_unterminated_block():
    """В оборванном монологе ответа нет — отдавать пользователю нечего."""
    from neftegaz.agent.llm import strip_reasoning

    assert strip_reasoning("Начало.\n<think>раздумья оборвались") == "Начало."


# ── переупорядочивание выдачи ──────────────────────────────────────────────
# Векторная близость меряет «про то же самое», а не «содержит ответ»: на
# корпусе EIA глоссарий с определением ОЭСР обходил таблицу с прогнозом,
# потому что в нём те же слова идут сплошной прозой. Замер до/после на
# четырёх вопросах: фрагментов с числами в top-5 было 10/20, стало 18/20.


def test_digit_ratio_separates_table_from_prose():
    from neftegaz.rag.store import digit_ratio

    table = "103.44 105.01 107.75 108.20 103.86 95.26 101.29"
    prose = "OECD means the Organization for Economic Cooperation and Development"
    assert digit_ratio(table) > digit_ratio(prose)


def test_rerank_prefers_numeric_chunk_at_equal_similarity():
    from neftegaz.rag.store import rerank_score

    table = "Brent 92.70 95.10 98.40 101.20 млн барр./сут 103.4 105.0"
    glossary = "OECD = Organization for Economic Cooperation and Development, including Australia"
    assert rerank_score(table, 0.70) > rerank_score(glossary, 0.70)


def test_rerank_penalty_does_not_discard_tables_with_footnotes():
    """★Главный тест: сноска под таблицей не должна топить саму таблицу.

    Фильтр по маркерам выбрасывал 10% корпуса, и среди выброшенного были
    таблицы, к которым прилипла строка про independent rounding. Поэтому
    маркер лишь понижает ранг, а плотность чисел это понижение перебивает.
    """
    from neftegaz.rag.store import rerank_score

    table_with_footnote = (
        "8.13 8.59 8.78 12.50 12.53 13.25 1,232 1,279 1,267 1,236 1,205 "
        "Totals may not equal sum of components due to independent rounding."
    )
    pure_glossary = (
        "Consumption of petroleum by the OECD countries is the same as petroleum "
        "product supplied, defined in the glossary of the EIA Petroleum Supply Monthly."
    )
    assert rerank_score(table_with_footnote, 0.70) > rerank_score(pure_glossary, 0.72)


def test_rerank_adjustment_is_small_enough_not_to_promote_junk():
    """Поправка меняет порядок внутри релевантного, а не тащит постороннее.

    Разрыв между профильным (0.700) и посторонним (0.452) — 0.248; поправка
    ограничена 0.06, то есть перекрыть этот разрыв она не может.
    """
    from neftegaz.rag.store import DATA_BONUS, rerank_score

    dense_but_offtopic = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
    assert rerank_score(dense_but_offtopic, 0.452) < 0.452 + DATA_BONUS + 1e-9
    assert rerank_score(dense_but_offtopic, 0.452) < 0.700


# ── шкала оси ──────────────────────────────────────────────────────────────
# ★Второй признак, добавленный 24.08.2026 после того, как первый оказался
# неспособен разделить ТРИ класса. Плотность цифр упорядочивает «глоссарий —
# таблица», но подпись к графику по ней неотличима от таблицы: метки осей это
# сплошные числа. Модель тогда вывела «добыча до 14.5 млн барр./сут» из
# ПОТОЛКА ОСИ Y. Замер после правки: страниц-шкал в top-5 было 4, стало 0.


def test_axis_scale_detects_chart_ticks():
    from neftegaz.rag.store import axis_scale

    assert axis_scale("U.S. crude oil production 11.5 12.0 12.5 13.0 13.5 14.0 14.5 million b/d")


def test_axis_scale_spares_year_headers():
    """★Ряд лет — арифметическая прогрессия, но это ЗАКОННЫЙ заголовок таблицы.

    Без этого исключения признак топил бы ровно те страницы, ради которых
    заведён: у каждой годовой таблицы STEO шапка вида 2024 2025 2026 2027.
    """
    from neftegaz.rag.store import axis_scale

    assert not axis_scale("Table 3c 2024 2025 2026 2027 2028 World total 103.44 105.01")


def test_axis_scale_spares_row_numbering():
    from neftegaz.rag.store import axis_scale

    assert not axis_scale("1 2 3 4 5 6 Commercial Sector 12.50 8.13 8.59")


def test_axis_scale_spares_real_data_series():
    """Настоящий ряд значений прогрессий не образует — замерено на корпусе."""
    from neftegaz.rag.store import axis_scale

    assert not axis_scale("World total 103.44 105.01 107.75 108.20 103.86 95.26 101.29")


def test_axis_page_ranks_below_table_despite_more_digits():
    """★Главный тест правки: подпись к графику ПРОИГРЫВАЕТ таблице.

    Числа взяты близко к замеренным: у подписи плотность цифр выше (метки осей
    — сплошные числа) и косинус чуть выше, у таблицы плотность ниже. До правки
    подпись выигрывала; теперь штраф за шкалу перевешивает бонус за плотность.
    """
    from neftegaz.rag.store import rerank_score

    axis_page = "Data source: U.S. Energy Information Administration 11.5 12.0 12.5 13.0 13.5 14.0"
    data_table = (
        "World total 103.44 105.01 107.75 108.20 103.86 95.26 101.29 107.14 "
        "OPEC total 32.15 33.08 34.11 million barrels per day"
    )
    assert rerank_score(data_table, 0.732) > rerank_score(axis_page, 0.747)


# ── русско-английский мостик к поиску по словам ────────────────────────────
# ★Этот мостик — единственное, что отделяет гибридный поиск от тихого отказа:
# корпус английский, вопросы русские, и без перевода терминов BM25 законно
# вернул бы пустоту, а выдача осталась бы прежней. Отказ выглядел бы не как
# ошибка, а как «гибрид не помог», поэтому мостик проверяется отдельно.


def test_bridge_translates_domain_terms():
    from neftegaz.rag.keyword import expand_query

    words = expand_query("Какая добыча нефти в США?")
    assert "production" in words
    assert "united" in words and "states" in words
    assert "crude" in words


def test_bridge_matches_by_word_beginning_not_whole_word():
    """Русский склоняется: «нефть», «нефти», «нефтью» обязаны попасть одинаково."""
    from neftegaz.rag.keyword import expand_query

    for form in ("нефть", "нефти", "нефтью", "нефтяной"):
        assert "oil" in expand_query(form), form


def test_bridge_passes_latin_through():
    """Вопрос может быть смешанным; терять уже подходящее слово нельзя."""
    from neftegaz.rag.keyword import expand_query

    words = expand_query("прогноз Brent")
    assert "brent" in words
    assert "forecast" in words


def test_bridge_drops_repeats_but_keeps_order():
    from neftegaz.rag.keyword import expand_query

    words = expand_query("нефть и нефтепродукты, добыча нефти")
    assert len(words) == len(set(words))
    assert words.index("oil") < words.index("production")


def test_tokenizer_drops_numbers():
    """★Числа — плохие поисковые слова: «13.28» стоит в десятках разных таблиц."""
    from neftegaz.rag.keyword import tokenize

    assert tokenize("United States 13.28 13.51") == ["united", "states"]


# ── BM25 ───────────────────────────────────────────────────────────────────


def _index(*documents):
    from neftegaz.rag.keyword import BM25Index

    index = BM25Index()
    for document in documents:
        index.add(document)
    index.finalise()
    return index


def test_bm25_finds_the_document_that_has_the_words():
    index = _index(
        "United States crude oil production 13.28 13.51",
        "Natural gas storage in Europe remained above the five-year average",
    )
    ranked = index.rank(["crude", "oil", "production"], limit=5)
    assert ranked[0][0] == 0


def test_bm25_ignores_words_that_are_everywhere():
    """Слово из каждого документа не решает исход, редкое — решает.

    Корпус взят большим намеренно: «вес около нуля» — свойство большого
    корпуса, а не арифметическое тождество. На трёх документах у вездесущего
    слова вес 0.134, и порог, поставленный без этой оговорки, ловил бы не
    свойство, а размер выборки.
    """
    index = _index(*[f"oil report word{n}" for n in range(200)])
    assert index.idf("oil") < 0.01  # в каждом из двухсот
    assert index.idf("word7") > 4.0  # ровно в одном
    assert index.idf("word7") > 100 * index.idf("oil")


def test_bm25_prefers_the_shorter_document_at_equal_evidence():
    """Одно совпадение в короткой строке говорит о ней больше, чем в длинной."""
    index = _index("crude oil production", "crude " + "filler " * 200)
    ranked = dict(index.rank(["crude"], limit=5))
    assert ranked[0] > ranked[1]


def test_bm25_returns_nothing_rather_than_everything_on_an_empty_query():
    index = _index("United States crude oil production")
    assert index.rank([], limit=5) == []


def test_bm25_survives_an_empty_corpus():
    """Пустой корпус — не повод для деления на ноль в глубине."""
    from neftegaz.rag.keyword import BM25Index

    index = BM25Index()
    index.finalise()
    assert index.rank(["oil"], limit=5) == []


# ── слияние двух выдач ─────────────────────────────────────────────────────


def test_agreement_of_two_methods_beats_certainty_of_one():
    """★Проверяется САМ ВЫБОР RRF_K, а не то, что формула считается.

    Замысел слияния: документ, стоящий в обоих списках в середине, обязан
    обгонять чемпиона одного списка. Именно поэтому смягчитель большой — при
    маленьком RRF_K первое место весило бы непропорционально много, и слияние
    выродилось бы в «побеждает первый хоть где-то», то есть в отсутствие
    слияния. Тест сломается, если константу тронут не подумав.
    """
    from neftegaz.rag.store import RRF_K

    both_in_the_middle = 1.0 / (RRF_K + 5) + 1.0 / (RRF_K + 5)
    champion_of_one = 1.0 / (RRF_K + 1)
    assert both_in_the_middle > champion_of_one


def test_disqualification_agrees_with_the_penalty_it_mirrors():
    """★Два ответа на один вопрос обязаны совпадать.

    Векторная ветвь вычитает из косинуса, словесная — не пускает вовсе, но
    решают они одно и то же: несёт ли фрагмент данные. Разъедутся — отказ
    ничем себя не проявит, кроме странной выдачи через полгода.
    """
    from neftegaz.rag.store import carries_no_data, rerank_score

    axis_page = "Data source: EIA 11.5 12.0 12.5 13.0 13.5 14.0"
    data_row = "United States 13.28 13.51 13.78 13.77 13.63"

    assert carries_no_data(axis_page)
    assert not carries_no_data(data_row)
    # тот же признак в другой валюте: наказан — значит и дисквалифицирован
    assert rerank_score(axis_page, 0.70) < 0.70


# ── маршрут прогноза обязан спрашивать корпус ──────────────────────────────


def _graph_with_stub_nodes(monkeypatch, route: str, report_hits: list):
    """Собрать граф из подменённых узлов: проверяем РЁБРА, а не содержимое.

    Узлы подменяются до сборки, потому что LangGraph забирает функцию в момент
    add_node: подмена после компиляции не дошла бы до графа и тест был бы
    зелёным при любом устройстве рёбер.
    """
    from neftegaz.agent import graph as G

    monkeypatch.setattr(G, "node_route", lambda state: {"route": route})
    monkeypatch.setattr(G, "node_forecast", lambda state: {"forecast_text": "РАСЧЁТ"})
    monkeypatch.setattr(
        G,
        "node_retrieve",
        lambda state: {"report_hits": list(report_hits), "used_reports": bool(report_hits)},
    )
    monkeypatch.setattr(G, "node_web", lambda state: {"web_hits": ["веб"], "used_web": True})
    monkeypatch.setattr(
        G,
        "node_answer",
        lambda state: {"answer": "|".join(sorted(k for k in state if k != "question"))},
    )
    return G.build_graph()


def test_forecast_route_still_consults_the_report_corpus(monkeypatch):
    """★Ветка прогноза обязана пройти через корпус отчётов.

    Дефект, ради которого написан тест: вопрос «какой прогноз EIA по Brent на
    2027 год» классификатор относит к прогнозу, и граф уходил прямо в свой
    расчёт, отвечая «в базе отчётов ничего не найдено». Между тем поиск по
    тому же вопросу возвращает фрагменты с близостью 0.72 при пороге 0.55, и
    нужная таблица STEO в корпусе есть.

    Требование 2.4 говорит о приоритете источников, а не о выборе одного из
    них: собственный расчёт дополняет отчёты и не заменяет их.
    """
    app = _graph_with_stub_nodes(monkeypatch, "forecast", ["ф1", "ф2", "ф3", "ф4", "ф5"])
    out = app.invoke({"question": "Какой прогноз EIA по ценам на Brent в 2027 году?"})

    assert out.get("used_reports") is True, "ветка прогноза прошла мимо корпуса отчётов"
    assert out.get("forecast_text") == "РАСЧЁТ", "расчётный модуль перестал работать"


def test_forecast_route_falls_through_to_web_when_the_corpus_is_thin(monkeypatch):
    """Правило приоритета источников не отменяется веткой прогноза.

    Отрицательный контроль к тесту выше: если бы корпус подключили в обход
    условия `_after_retrieve`, веб не запускался бы никогда и разница между
    полной и пустой выдачей корпуса стала бы невидимой.
    """
    app = _graph_with_stub_nodes(monkeypatch, "forecast", [])
    out = app.invoke({"question": "Спрогнозируй Brent на 90 дней"})

    assert out.get("used_web") is True, "пустой корпус не привёл к веб-поиску"


def test_industry_route_does_not_run_the_calculation(monkeypatch):
    """Второй отрицательный контроль: ветки не слились в одну.

    Починка «пусть прогноз тоже ходит в корпус» соблазняет свести граф к
    одному пути, где считается всё и всегда. Тогда на отраслевой вопрос без
    просьбы о прогнозе в ответ поедет ARIMA, а расчёт — самый дорогой узел.
    """
    app = _graph_with_stub_nodes(monkeypatch, "industry", ["ф1", "ф2", "ф3"])
    out = app.invoke({"question": "Что происходит с добычей ОПЕК+?"})

    assert out.get("used_reports") is True
    assert "forecast_text" not in out, "расчёт запустился на вопросе, где его не просили"


# ── шапка колонок таблицы ──────────────────────────────────────────────────
# ★Дефект, ради которого это написано: на вопросе о добыче ОПЕК+ модель ответила,
# что видит строку «OPEC members subject to OPEC+ agreements» с рядом чисел, но
# не видит, какому периоду какое число принадлежит, и оговорила единицы как
# предположение. Заголовок таблицы фрагмент нёс, шапку колонок — нет.

_HEADER_STREAM = (
    "Table 3c. World Petroleum and Other Liquid Fuels Production (million barrels per day)\n"
    "U.S. Energy Information Administration | Short-Term Energy Outlook - July 2026\n"
    "Q1 Q2 Q3 Q4 Q1 Q2 Q3 Q4 Q1 Q2 Q3 Q4 2025 2026 2027\n"
    "Petroleum and other liquid fuels production (a)\n"
    "OPEC members subject to OPEC+ agreements (d) ....... 21.55 21.96 22.38 22.78 20.60"
    " 13.75 17.44 21.51 22.48 22.71 22.83 22.92 22.17 18.33 22.74\n"
    "United States ...................................... 22.75 23.49 24.10 24.09 23.71"
    " 24.31 24.25 24.47 24.45 24.89 24.91 25.03 23.61 24.19 24.82\n"
    "Refinery capacity (e) ........................ 18.4 18.4 18.4 18.5 18.5 18.6\n"
)
_COLUMNS = "Q1 Q2 Q3 Q4 Q1 Q2 Q3 Q4 Q1 Q2 Q3 Q4 2025 2026 2027"


def _row_named(chunks: list[dict], label: str) -> dict:
    """Фрагмент-строка, несущая эту подпись.

    Поиск по вхождению, а не по началу текста: у первой строки раздела в
    подпись затягивается предыдущая строка-заголовок раздела («Petroleum and
    other liquid fuels production (a)»), потому что граница подписи — последнее
    число предыдущей строки, а чисел в ней нет.
    """
    found = [c for c in chunks if c["kind"] == "row" and label in c["text"]]
    assert found, f"строка {label!r} не стала отдельным фрагментом"
    return found[0]


def test_row_carries_the_column_header_not_only_the_caption():
    """Строка несёт и заголовок таблицы, и привязку чисел к столбцам.

    Привязка приходит парами «столбец = значение», а не двумя отдельными
    списками: раньше здесь проверялось наличие ряда «Q1 Q2 Q3 …» дословно, и
    этого было мало — ряд названий сам по себе оставлял модели сопоставление по
    счёту. Требование то же («числа не должны остаться безымянными»), проверка
    строже: у каждого числа назван ЕГО столбец.
    """
    chunks = chunk_pages([{"page": 31, "text": _HEADER_STREAM}], size=200, overlap=40)
    row = _row_named(chunks, "OPEC members subject")

    assert "World Petroleum" in row["context"], "потерян заголовок таблицы"
    context = row["context"]
    assert "Q1 = 21.55" in context, "первое число не привязано к своему столбцу"
    assert "2027 = 22.74" in context, "последнее число не привязано к своему столбцу"
    assert context.count("=") == len(_COLUMNS.split()), "привязаны не все столбцы"


def test_pairs_are_refused_when_the_counts_do_not_match():
    """★Отрицательный контроль: неполная строка пар не получает.

    У «Refinery capacity» шесть значений при пятнадцати столбцах. Любая пара
    здесь была бы выдумкой — и выдумкой правдоподобной, потому что все числа
    настоящие. Строка остаётся с одним заголовком таблицы.
    """
    chunks = chunk_pages([{"page": 31, "text": _HEADER_STREAM}], size=200, overlap=40)
    row = _row_named(chunks, "Refinery capacity")

    assert "=" not in row["context"], "пары построены при несовпадении длин"
    assert "World Petroleum" in row["context"], "потерян заголовок таблицы"


def test_pairs_keep_the_dash_as_a_value():
    """Прочерк — утверждение «данных нет», а не пропуск.

    Выбросив его молча, мы сдвинули бы все следующие значения на столбец влево:
    ровно та ошибка, ради которой пары и написаны, только изнутри.
    """
    assert row_pairs("Alpha ... 1.0 - 3.0", "Q1 Q2 Q3") == "Alpha: Q1 = 1.0, Q2 = -, Q3 = 3.0"


def test_running_header_is_not_mistaken_for_the_column_header():
    """★Отрицательный контроль: шапка узнаётся по составу, а не по положению.

    Сразу под заголовком таблицы стоит колонтитул, и в нём есть «July 2026».
    Правило «строка сразу под заголовком» взяло бы его, и фрагмент получил бы
    вместо периодов название бюллетеня — ошибку, которую нечем заметить.
    """
    from neftegaz.rag.chunking import is_column_header

    assert not is_column_header(
        "U.S. Energy Information Administration | Short-Term Energy Outlook - July 2026"
    )
    assert is_column_header(_COLUMNS)

    chunks = chunk_pages([{"page": 31, "text": _HEADER_STREAM}], size=200, overlap=40)
    assert "Short-Term Energy Outlook" not in _row_named(chunks, "United States")["context"]


def test_data_row_is_not_mistaken_for_a_column_header():
    """Числа таблицы — не обозначения периодов, даже когда их много."""
    from neftegaz.rag.chunking import is_column_header

    assert not is_column_header("21.55 21.96 22.38 22.78 20.60")
    assert not is_column_header("- - 1.38 - -")


def test_two_dates_in_a_line_are_not_a_column_header():
    """Порог в три обозначения: пара дат встречается и в прозе."""
    from neftegaz.rag.chunking import is_column_header

    assert not is_column_header("2025 2026")
    assert is_column_header("2025 2026 2027")


def test_column_header_of_a_distant_table_is_not_borrowed():
    """★Чужая шапка врёт незаметнее чужого заголовка.

    Заголовок чужой таблицы виден в ответе и вызывает вопрос. Чужая шапка
    молча привяжет числа к периодам другой таблицы, и ответ будет выглядеть
    обычным. Правило удалённости у них поэтому одно.
    """
    from neftegaz.rag.chunking import (
        MAX_CAPTION_DISTANCE,
        caption_positions,
        header_positions,
        table_context_before,
    )

    stream = _HEADER_STREAM + "x" * (MAX_CAPTION_DISTANCE + 100)
    captions = caption_positions(stream)
    headers = header_positions(stream, captions)

    assert _COLUMNS in table_context_before(captions, headers, 200)
    assert table_context_before(captions, headers, len(stream) - 1) == ""


def test_context_already_present_in_the_body_is_not_repeated():
    """Удвоенные слова размывают эмбеддинг; части взвешиваются по отдельности."""
    from neftegaz.rag.chunking import context_outside

    assert context_outside(f"хвост {_COLUMNS} хвост", f"Table 3c. X\n{_COLUMNS}") == "Table 3c. X"
    assert (
        context_outside("числа без слов", f"Table 3c. X\n{_COLUMNS}") == f"Table 3c. X\n{_COLUMNS}"
    )


def test_answering_prompt_shows_the_table_context_and_keeps_text_verbatim():
    """★Контекст обязан дойти до ОТВЕЧАЮЩЕЙ модели, а не только до эмбеддинга.

    Раньше context участвовал лишь в вычислении вектора. Найденный фрагмент
    доезжал до модели голым рядом чисел — то есть починка нарезки без этого
    шага не изменила бы ответ ни на слово.
    """
    from neftegaz.agent.graph import _format_report_context
    from neftegaz.rag.store import Hit

    row = "OPEC members subject to OPEC+ agreements (d) ....... 21.55 21.96"
    hit = Hit(
        text=row,
        score=0.71,
        source_name="EIA STEO",
        date="июль 2026",
        page=31,
        page_end=31,
        context=f"Table 3c. World Petroleum\n{_COLUMNS}",
    )
    rendered = _format_report_context([hit])

    assert "Table 3c. World Petroleum" in rendered
    assert _COLUMNS in rendered
    assert row in rendered, "текст фрагмента обязан ехать дословно"
    assert f"{_COLUMNS}\n{row}" not in rendered, "контекст склеился с текстом цитаты"


def test_last_value_of_a_row_is_not_dropped():
    """★Последний столбец строки — годовой, то есть прогноз на дальний год.

    Требование «за каждым значением идёт пробел» отрезало ровно одно значение:
    то, за которым стоит перевод строки. Замерено на отчёте за июль 2026 — 903
    строки из 938. Спрашивают именно про этот столбец.
    """
    from neftegaz.rag.chunking import table_rows

    stream = "United States ....... 22.75 23.49 24.10 24.09 24.82\nСледующая строка\n"
    spans = table_rows(stream)
    assert spans, "строка таблицы не распознана"
    assert stream[spans[0][0] : spans[0][1]].rstrip().endswith("24.82")


def test_narrow_row_does_not_get_the_wide_header():
    """★Шапка поверяется по КАЖДОЙ строке, а не только по первой строке таблицы.

    Пятнадцать периодов над шестью числами читаются как «данных в конце нет»,
    а на деле смещены все. Заголовок таблицы при этом остаётся: к ширине он
    отношения не имеет.
    """
    chunks = chunk_pages([{"page": 31, "text": _HEADER_STREAM}], size=200, overlap=40)
    narrow = _row_named(chunks, "Refinery capacity")

    assert "World Petroleum" in narrow["context"], "заголовок таблицы потерян зря"
    assert _COLUMNS not in narrow["context"], "широкая шапка приклеена к узкой строке"


def test_header_glued_to_the_preceding_sentence_is_still_found():
    """★Извлекатель PDF склеивает шапку с концом предыдущей фразы.

    В отчётах это выглядит как «…Energy Information Administration.Q1 Q2 Q3 Q4
    …». Проверка по строке целиком такую шапку не видит, хотя она есть, — ищем
    РЯД обозначений, а не строку.
    """
    from neftegaz.rag.chunking import column_header_after

    stream = (
        "Table 2. Energy Prices (dollars per barrel)\n"
        "Weather forecasts from National Oceanic and Atmospheric Administration"
        " and Energy Information Administration.Q1 Q2 Q3 2025\n"
        "West Texas Intermediate Spot Average ....... 71.85 64.63 65.78 65.40\n"
    )
    assert column_header_after(stream, 0) == "Q1 Q2 Q3 2025"


def test_chunk_never_spans_a_table_caption():
    """★Фрагмент не пересекает заголовок таблицы.

    Без этого правила окно начинается в сносках одной таблицы и продолжается
    числами следующей, а подпись ему выбирается по НАЧАЛУ — и приезжает чужая.
    Замер по корпусу до починки: 173 фрагмента из 9456 несли подпись, не
    совпадающую с таблицей, чьи цифры в них лежат.

    Проверка двусторонняя по построению: страница составлена так, что при
    старом правиле граница ОБЯЗАНА была пройти сквозь заголовок — хвост первой
    таблицы длиннее шага окна, но короче его размера.
    """
    from neftegaz.rag.chunking import TABLE_CAPTION, chunk_pages

    first = "Table 1. Crude Oil Prices\n" + "Brent Spot Average 75.83 68.01 69.00\n" * 12
    tail = "Notes: values are annual averages.\n" * 6
    second = "Table 2. Energy Prices\n" + "Propane Residential 2.71 2.48 2.64\n" * 40
    pages = [{"page": 1, "text": first + tail + second}]

    chunks = chunk_pages(pages, size=1200, overlap=200)
    windows = [c for c in chunks if c.get("kind") != "row"]
    assert len(windows) > 1, "страница должна разрезаться, иначе тест ничего не проверяет"

    for chunk in windows:
        text = chunk["text"]
        offset = len(text) - len(text.lstrip())
        for match in TABLE_CAPTION.finditer(text):
            # Заголовок внутри допустим ровно в одном случае: фрагмент с него
            # начат. ★Проверяется КАЖДОЕ вхождение, а не первое: у фрагмента,
            # который начинается своим заголовком и втягивает следующий,
            # первое вхождение стоит на месте и ничего не ловит.
            assert match.start() == offset, (
                f"фрагмент пересёк заголовок {match.group()!r} на позиции "
                f"{match.start()}, начавшись с {text[:40]!r}"
            )


def test_context_fits_the_budget_and_says_when_it_was_cut():
    """★Контекст держится в потолке знаков, и усечение помечено.

    Замер 26.08: на вопросе про сопоставление прогноза с котировками сервер
    модели ответил 500 context_length_exceeded, и агент выродился в свалку
    сырых фрагментов. Проверка двусторонняя: с бюджетом впору — не режем и
    метки не ставим; заведомо больше бюджета — режем и метку ставим.
    """
    from neftegaz.agent.graph import (
        TRUNCATION_MARK,
        _format_report_context,
    )
    from neftegaz.config import settings

    # Бюджет читается из настроек: он стал правимым параметром хода, и тест
    # обязан спрашивать действующее значение, а не помнить прежнюю константу.
    report_budget_chars = settings.report_budget_chars

    class FakeHit:
        def __init__(self, text, index):
            self.text = text
            self.source_name = "EIA STEO"
            self.date = "июль 2026"
            self.page = 10 + index
            self.page_end = 10 + index
            self.score = 0.9 - index / 100
            self.context = "Table 2. Energy Prices"

    small = [FakeHit("Brent Spot Average 75.83 68.01\n" * 4, i) for i in range(3)]
    rendered = _format_report_context(small)
    assert len(rendered) <= report_budget_chars
    assert TRUNCATION_MARK not in rendered
    assert rendered.count("[фрагмент:") == 3, "короткие фрагменты обрезать не за что"

    huge = [FakeHit("x" * 4000, i) for i in range(8)]
    rendered = _format_report_context(huge)
    assert len(rendered) <= report_budget_chars + len(TRUNCATION_MARK) * 8
    assert TRUNCATION_MARK in rendered, "усечение обязано быть видно модели"
    assert rendered.count("[фрагмент:") < 8, "часть фрагментов обязана отсеяться"
    # ★Пустого контекста быть не должно ни при каком бюджете: он превратил бы
    # ответ по источникам в ответ по памяти модели.
    assert _format_report_context([FakeHit("y" * 50000, 0)]).strip()


# ── детерминизм индекса и выдачи ───────────────────────────────────────────
# ★Добавлено 27.08.2026. Оба отказа ниже НЕ ПАДАЮТ и не видны в обычных тестах:
# они возвращают правдоподобный результат, который в следующий раз оказывается
# другим. Ловятся только тем, что проверяется прямо.


def test_chunk_id_is_derived_from_content_not_random():
    """Идентификатор фрагмента обязан быть один и тот же при каждой сборке.

    Со случайным идентификатором две сборки одного корпуса давали разные
    индексы, повторная загрузка отчёта добавляла копии вместо замены, а равные
    оценки нечем было развязать устойчиво.
    """
    from neftegaz.rag.store import chunk_id

    chunk = {
        "source_name": "EIA STEO",
        "date": "июль 2026",
        "kind": "row",
        "page": 31,
        "page_end": 31,
        "start": 1200,
        "text": "United States ... 13.28",
    }
    assert chunk_id(chunk) == chunk_id(dict(chunk)), "идентификатор не воспроизводится"

    other = dict(chunk, page=32)
    assert chunk_id(chunk) != chunk_id(other), "разные страницы дали один идентификатор"

    same_text_other_report = dict(chunk, date="июнь 2026")
    assert chunk_id(chunk) != chunk_id(same_text_other_report), "разные отчёты слились"


def test_equal_scores_are_broken_deterministically():
    """При равных оценках порядок задаётся содержанием, а не ответом хранилища.

    Порядок точек с одинаковым косинусом хранилищем не оговорён. Устойчивая
    сортировка сохраняет входной порядок — но только он и не устойчив.
    """
    from neftegaz.rag.store import _identity_of

    one = {"date": "июнь 2026", "source_name": "EIA STEO", "page": 9, "kind": "row", "text": "a"}
    two = {"date": "июль 2026", "source_name": "EIA STEO", "page": 3, "kind": "row", "text": "a"}

    assert _identity_of(one) == _identity_of(dict(one)), "ключ не воспроизводится"
    assert _identity_of(one) != _identity_of(two), "разные фрагменты дали один ключ"

    # Главное: порядок не зависит от того, в каком порядке пришли записи.
    assert sorted([one, two], key=_identity_of) == sorted([two, one], key=_identity_of)

    # ★А вот чего ключ НЕ обещает — хронологии. «июль» встаёт раньше «июня»,
    # потому что это сравнение строк. Проверяется явно, чтобы обещание не
    # выросло в комментариях само собой: устойчивость есть, хронологии нет.
    assert _identity_of(two) < _identity_of(one), "поведение ключа изменилось — сверить docstring"
