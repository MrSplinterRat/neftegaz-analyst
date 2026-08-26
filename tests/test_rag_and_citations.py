"""Tests for chunking, citation formatting, routing and the web filter.

The citation tests are the load-bearing ones. A wrong number in a forecast is a
bad forecast; a wrong or invented citation makes an unverified claim look
verified, which is the failure this product exists to prevent.
"""

from __future__ import annotations

import pytest

from neftegaz.agent.graph import parse_horizon_days, parse_supply_change
from neftegaz.rag.chunking import chunk_document, chunk_pages
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
        [{"page": 3, "text": "z" * 40}], source_name="EIA STEO", doc_date="июль 2026", size=40, overlap=0
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

    rows = [c for c in split([{"page": 7, "text": _ROW_STREAM}], size=200, overlap=40)
            if c["kind"] == "row" and c["text"].startswith("United States")]
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
    }
    assert format_claim(claim) == (
        "Спрос вырастет на 1.2 млн барр./сут. [Отчёт OPEC MOMR, март 2025, с. 14]"
    )


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
        title="Reuters", url="https://reuters.com/x", snippet="текст", domain="reuters.com", preferred=True
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
    assert index.idf("oil") < 0.01     # в каждом из двухсот
    assert index.idf("word7") > 4.0    # ровно в одном
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
        G, "node_retrieve",
        lambda state: {"report_hits": list(report_hits), "used_reports": bool(report_hits)},
    )
    monkeypatch.setattr(G, "node_web", lambda state: {"web_hits": ["веб"], "used_web": True})
    monkeypatch.setattr(
        G, "node_answer",
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
