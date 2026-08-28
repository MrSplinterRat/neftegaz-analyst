"""Tests for the differential cross-check.

The verdict logic lives in ``compare_pages``, which takes two strings — so the
interesting cases are written as text, not as PDFs. That is deliberate: a rule
that can only be exercised through a real document cannot be exercised at all
on the damage we have not met yet, and the damage we have not met yet is the
whole point of the module.
"""

from __future__ import annotations

import os

import pytest

from neftegaz.rag.crosscheck import (
    AGREE,
    DIVERGE,
    ORDER,
    TOKENIZE,
    VERDICT_ORDER,
    compare_pages,
    compare_pages_multi,
    crosscheck_directory,
    crosscheck_pdf,
    numbers_in,
    reduce_to_comparable,
)

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "reports")
ONE_REPORT = os.path.join(REPORTS_DIR, "EIA_STEO_2026-07.pdf")


# ── выделение чисел ────────────────────────────────────────────────────────


def test_thousands_separator_and_leading_plus_are_normalised():
    assert numbers_in("Total 1,234.5 and +7") == ["1234.5", "7"]


def test_numbers_glued_to_words_are_not_numbers():
    """Q1 and Table3 are labels; counting them would make noise look like data."""
    assert numbers_in("Q1 Table3 ref2026a") == []


def test_negative_numbers_survive():
    assert numbers_in("change -0.4 mb/d") == ["-0.4"]


def test_order_is_preserved():
    assert numbers_in("1 2 3") == ["1", "2", "3"]


# ── вердикты ───────────────────────────────────────────────────────────────


def test_identical_pages_agree():
    assert compare_pages("brent 82.4 wti 78.1", "brent 82.4 wti 78.1", 1).verdict == AGREE


def test_same_numbers_in_a_different_order_is_a_reading_order_finding():
    """Nothing is lost, the caption moved — our known EIA case."""
    diff = compare_pages("82.4 78.1", "78.1 82.4", 1)
    assert diff.verdict == ORDER
    assert diff.lost == 0


def test_glued_axis_labels_are_tokenisation_not_loss():
    """poppler renders two neighbouring 2024 labels as 20242024."""
    diff = compare_pages("20242024 82.4", "2024 2024 82.4", 1)
    assert diff.verdict == TOKENIZE


def test_tokenisation_verdict_survives_a_shuffle():
    """The digit multiset ignores both the breaks and the order, by construction."""
    assert compare_pages("00 99 88", "9 8 0 9 8 0", 1).verdict == TOKENIZE


def test_a_missing_number_is_divergence_not_tokenisation():
    diff = compare_pages("82.4", "82.4 48", 1)
    assert diff.verdict == DIVERGE
    assert diff.only_b == ["48"]
    assert diff.only_a == []
    assert diff.lost == 1


def test_a_substituted_digit_is_divergence():
    """The font-decoding case: same shape of page, one wrong character."""
    diff = compare_pages("82.4", "83.4", 1).verdict
    assert diff == DIVERGE


def test_empty_page_on_one_side_is_divergence():
    diff = compare_pages("", "82.4", 1)
    assert diff.verdict == DIVERGE
    assert diff.only_b == ["82.4"]


def test_two_empty_pages_agree():
    """A page with no numbers on either side is not a finding."""
    assert compare_pages("", "", 1).verdict == AGREE


# ── о чём пара вправе судить ───────────────────────────────────────────────


def test_pair_with_a_non_positional_reader_judges_only_the_digits():
    """★Иначе третий читатель заглушил бы первых двух, а не дополнил их.

    pypdf выдаёт текст в порядке операторов потока, поэтому ORDER против него
    зажигается почти на каждой странице (замерено: 322 из 469). Признак,
    срабатывающий всегда, не различает ничего, но по правилу «сводный вердикт —
    худший из попарных» тянет вниз всю страницу и прячет настоящие находки
    позиционной пары. Поэтому паре с таким читателем оставлен ровно один
    вопрос — тот, на который он отвечает лучше всех: те же ли это цифры.
    """
    assert reduce_to_comparable(ORDER, positional=False) == AGREE
    assert reduce_to_comparable(TOKENIZE, positional=False) == AGREE
    # ★Расхождение по цифрам обязано пережить сведение: ради него третий путь
    # и добавлен — он свидетельствует о ДЕКОДИРОВАНИИ, а не о сборке строк.
    assert reduce_to_comparable(DIVERGE, positional=False) == DIVERGE
    assert reduce_to_comparable(AGREE, positional=False) == AGREE


def test_pair_of_positional_readers_keeps_every_verdict():
    """Двое читают от положения на странице — значит вправе судить и о порядке."""
    for verdict in VERDICT_ORDER:
        assert reduce_to_comparable(verdict, positional=True) == verdict


# ── три читателя ───────────────────────────────────────────────────────────


def test_the_reader_that_disagrees_with_both_others_is_named():
    """★Ради этого и заведён третий путь: при двух известно лишь «кто-то неправ».

    Здесь двое видят 78.1, третий — 79.1. Указание на виновника не доказывает,
    что большинство право (на общем декодере оно ошибётся хором), но переводит
    находку из «странице нельзя верить» в «вот этот путь читает её не так», а
    это уже адрес для разбора. Пропадёт свойство — расхождение снова станет
    безадресным, и страницу придётся списывать целиком.
    """
    diff = compare_pages_multi(
        {"alpha": "82.4 78.1", "beta": "82.4 78.1", "gamma": "82.4 79.1"},
        page=1,
        primary=("alpha", "beta"),
    )
    assert diff.outlier == "gamma"
    assert diff.verdict == DIVERGE


def test_a_general_disagreement_names_nobody():
    """Все трое разошлись между собой — большинства нет, и выдумывать его нельзя.

    Соблазн назвать «самого непохожего» здесь силён и вреден: выброс определён
    как «разошёлся с обоими, а те двое согласны», и без второго условия имя
    было бы догадкой, выданной за результат сверки.
    """
    diff = compare_pages_multi(
        {"alpha": "1", "beta": "2", "gamma": "3"},
        page=1,
        primary=("alpha", "beta"),
    )
    assert diff.outlier is None
    assert diff.verdict == DIVERGE


def test_all_pairs_are_kept_and_the_summary_is_the_worst_of_them():
    """Сводка выводима из пар, обратно — нет, поэтому хранятся все пары.

    ★Пример подобран так, что алфавит и порядок тяжести расходятся: по алфавиту
    худшим из {tokenize, diverge} оказался бы `tokenize`. Сводный вердикт обязан
    считаться по VERDICT_ORDER — иначе потеря цифр однажды тихо спрячется за
    склейкой подписей, и разница между «переставлено» и «утрачено» исчезнет
    ровно там, где она дороже всего.
    """
    diff = compare_pages_multi(
        # alpha и beta различаются только членением тех же цифр; gamma вдобавок
        # не видит 82.4 — это уже потеря.
        {"alpha": "20242024 82.4", "beta": "2024 2024 82.4", "gamma": "2024 2024"},
        page=1,
        primary=("alpha", "beta"),
    )
    assert set(diff.pairs) == {"alpha|beta", "alpha|gamma", "beta|gamma"}
    assert diff.pairs["alpha|beta"] == TOKENIZE
    assert diff.pairs["alpha|gamma"] == DIVERGE
    assert diff.pairs["beta|gamma"] == DIVERGE
    assert diff.verdict == max(diff.pairs.values(), key=VERDICT_ORDER.index) == DIVERGE


# ── отчёт целиком ──────────────────────────────────────────────────────────


def test_missing_file_yields_a_note_not_an_exception():
    report = crosscheck_pdf("/nonexistent/nope.pdf")
    assert report.pages == []
    assert report.notes and "не ответил" in report.notes[0]
    assert report.counts()["pages"] == 0


def test_lost_counts_only_divergence():
    """Tokenisation must not inflate the loss figure — the digits are still there."""
    from neftegaz.rag.crosscheck import CrossCheckReport

    report = CrossCheckReport(path="x.pdf")
    report.pages = [
        compare_pages("20242024", "2024 2024", 1),   # tokenize
        compare_pages("82.4", "82.4 48", 2),         # diverge, one token
    ]
    counts = report.counts()
    assert counts[TOKENIZE] == 1
    assert counts[DIVERGE] == 1
    assert counts["numbers_lost"] == 1


def test_directory_scan_survives_a_missing_directory():
    assert crosscheck_directory("/nonexistent/dir") == []


# ── корпус ─────────────────────────────────────────────────────────────────


MAIN_PAIR = "pdf2xml|poppler"   # имена в ключе пары отсортированы, см. compare_pages_multi


@pytest.mark.skipif(not os.path.isfile(ONE_REPORT), reason="корпус не выложен")
def test_the_main_reading_path_mostly_agrees_on_a_real_report():
    """A sanity floor, not a quality claim.

    ★ПОРОГ НАМЕРЕННО НИЗКИЙ. Тест стережёт не качество извлечения, а
    работоспособность СВЕРКИ: если однажды один из читателей начнёт отдавать
    пустоту или мусор, доля согласных страниц рухнет, и это увидит тест, а не
    аналитик в цитате. Утверждать по этому числу «мы читаем хорошо» нельзя:
    согласие двух путей не доказывает правоты обоих.

    ★СМОТРИМ НА ПАРУ, А НЕ НА СВОДНЫЙ ВЕРДИКТ. Порог писался, когда путей было
    два, и мерил ровно основной тракт чтения. Сводный вердикт трёх путей — это
    худший из попарных, то есть в него подмешан и третий читатель, чья доля
    расхождений живёт своей жизнью: pypdf разбирает файл своим декодером, и
    добавить четвёртый путь значило бы снова сдвинуть это число, не тронув
    тракт, за которым тест поставлен следить. Порог по паре pdf2xml|poppler
    меряет то, ради чего написан: разлом основного чтения (замерено по корпусу
    353 согласных страницы из 469, то есть 0.75 против порога 0.5).
    """
    report = crosscheck_pdf(ONE_REPORT)
    # Оба читателя основного тракта обязаны были ответить: пара, которой нет,
    # не даст ни одного расхождения — и молчание прочиталось бы как согласие.
    assert {"pdf2xml", "poppler"} <= set(report.readers)
    verdicts = [page.pairs[MAIN_PAIR] for page in report.pages]
    assert verdicts
    assert verdicts.count(AGREE) / len(verdicts) > 0.5
