"""Сверка цитат в автоматической приёмке.

★ЗАЧЕМ ЭТО ЗДЕСЬ, А НЕ ТОЛЬКО В СКРИПТЕ. Проверяемость ссылок — предмет
поставки: продукт обещает, что названное число стои́т на названной странице.
Проверялось это обещание запуском скрипта руками. Проверка, которая случается
тогда, когда о ней вспомнили, ничем не отличается от проверки, которой нет:
отсутствие находок в ней означает, что её не запускали, а выглядит это ровно
как чистый результат.

★ГЛАВНОЕ ЗДЕСЬ — НЕ «ПРОВЕРКА ПРОШЛА», А «ПРОВЕРКА УМЕЕТ УПАСТЬ». Проверка,
проходящая всегда, не лучше отсутствующей, и мы уже дважды ловили себя на ней.
Поэтому сначала идут диверсии: число не с той страницы, число без ссылки,
число внутри ссылки с пометкой о спорном чтении. Каждая обязана быть поймана,
и каждая названа отдельным тестом — чтобы падение сказало, ЧТО именно перестало
ловиться.

★Корпус отчётов в публичный репозиторий не выкладывается (см. .gitignore).
Поэтому диверсии работают на подложном корпусе из двух страниц и идут ВСЕГДА, а
прогон по настоящим отчётам и настоящим демо-ответам добавляется сверху, когда
корпус на машине есть.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# scripts/ не пакет и в sys.path не входит: там лежат исполняемые сценарии.
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import check_citations  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
DEMO_DIR = PROJECT_ROOT / "demo"

# Подложный корпус: две страницы, чьё содержимое мы знаем наизусть.
PAGES = {
    ("июль 2026", 35): "U.S. crude oil production 13.28 13.51 13.78",
    ("июль 2026", 36): "World consumption 103.45",
}


def write_answer(directory: Path, body: str, question: str = "Что с добычей?") -> Path:
    """Демо-файл того же вида, что пишет прогон сценариев."""
    path = directory / "01-Проба.md"
    path.write_text(
        f"# Проба\n\n### Запрос\n\n> {question}\n\n### Ответ агента\n\n{body}\n---\n",
        encoding="utf-8",
    )
    return path


# ── диверсии: проверка обязана падать ──────────────────────────────────────


def test_a_number_that_is_not_on_the_cited_page_is_caught(tmp_path):
    answer = write_answer(
        tmp_path, "Добыча составила 14.99 [Отчёт EIA STEO, июль 2026, с. 35]."
    )
    code, totals = check_citations.run([answer], PAGES)
    assert code == 1
    assert totals["missing"] == 1


def test_a_number_with_no_citation_at_all_is_caught(tmp_path):
    answer = write_answer(tmp_path, "Добыча составила 13.28, и это наше мнение.")
    code, totals = check_citations.run([answer], PAGES)
    assert code == 1
    assert totals["unmarked"] == 1


def test_a_number_inside_a_marked_citation_is_still_checked(tmp_path):
    """★Пометка о спорном чтении не имеет права выводить ссылку из-под проверки.

    Ступень чтения дописывается в ссылку кодом, и ссылка после этого выглядит
    иначе. Если бы сверка перестала такие ссылки узнавать, вышло бы худшее из
    возможного: чем громче наша оговорка о странице, тем меньше её проверяют, а
    отчёт сверки остался бы зелёным.
    """
    answer = write_answer(
        tmp_path,
        "Добыча составила 14.99 "
        "[Отчёт EIA STEO, июль 2026, с. 35; ⚠ два пути чтения расходятся по цифрам].",
    )
    code, totals = check_citations.run([answer], PAGES)
    assert code == 1
    assert totals["missing"] == 1


def test_a_citation_to_a_page_outside_the_corpus_is_caught(tmp_path):
    answer = write_answer(tmp_path, "Запасы 12.34 [Отчёт EIA STEO, июль 2026, с. 99].")
    code, totals = check_citations.run([answer], PAGES)
    assert code == 1
    assert totals["missing"] == 1


def test_nothing_to_check_is_not_a_pass(tmp_path):
    """Третье состояние: «проверке не досталось работы» ≠ «проверено и чисто»."""
    answer = write_answer(tmp_path, "Данных по этому вопросу в отчётах нет.")
    code, totals = check_citations.run([answer], PAGES)
    assert code == 2
    assert totals["checked"] == 0


# ── и только теперь: чистое проходит ───────────────────────────────────────


def test_a_number_that_is_on_the_page_passes(tmp_path):
    answer = write_answer(
        tmp_path,
        "Добыча составила 13.28 [Отчёт EIA STEO, июль 2026, с. 35].",
    )
    code, totals = check_citations.run([answer], PAGES)
    assert code == 0
    assert totals == {"checked": 1, "missing": 0, "unmarked": 0, "web": 0}


def test_the_russian_spelling_of_a_number_is_the_same_number(tmp_path):
    answer = write_answer(
        tmp_path, "Добыча составила 13,28 [Отчёт EIA STEO, июль 2026, с. 35]."
    )
    code, _ = check_citations.run([answer], PAGES)
    assert code == 0


def test_a_marked_citation_with_a_true_number_passes(tmp_path):
    """Обратная сторона диверсии: пометка сама по себе не является находкой."""
    answer = write_answer(
        tmp_path,
        "Добыча составила 13.28 "
        "[Отчёт EIA STEO, июль 2026, с. 35; ⚠ два пути чтения расходятся по цифрам].",
    )
    code, totals = check_citations.run([answer], PAGES)
    assert code == 0
    assert totals["checked"] == 1


# ── прогон по настоящим отчётам, когда они есть на машине ──────────────────


@pytest.mark.skipif(
    not os.path.isdir(REPORTS_DIR) or not list(REPORTS_DIR.glob("*.pdf")),
    reason="корпус не выложен",
)
def test_the_shipped_demo_answers_survive_the_real_check():
    """★Это и есть приёмка поставки: демо-ответы против настоящих страниц.

    Обещание «откройте страницу 35 и увидите то же число» проверяется тут по
    самим PDF, а не по выдаче поиска: сверка с посредником подтвердила бы
    согласие модели с поиском и промолчала бы о том, что оба ошиблись вместе.
    """
    answers = sorted(DEMO_DIR.glob("*.md"))
    assert answers, "демо-ответов нет — проверять нечего"
    pages = check_citations.load_corpus()
    assert pages, "корпус на месте, но не прочитался"
    code, totals = check_citations.run(answers, pages)
    assert totals["checked"] > 0, "проверке не досталось ни одного числа со ссылкой"
    assert code == 0, (
        f"сверка цитат нашла расхождения: не подтверждено {totals['missing']}, "
        f"без ссылки {totals['unmarked']}"
    )
