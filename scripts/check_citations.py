#!/usr/bin/env python
"""Проверка цитат: стоит ли названное число на названной странице.

★ЗАЧЕМ ЭТО ОТДЕЛЬНЫМ ИНСТРУМЕНТОМ. Проверяемость ссылок — предмет поставки, а
не свойство, которое можно объявить. Модель, которой дали правильный фрагмент,
всё равно способна приписать ему число из соседнего или округлить его «для
удобства»; ровно эту разновидность отказа продукт и обязан не допускать, и
объявлять её отсутствующей без проверки — то же самое, что не проверять.

★ПРОВЕРКА НЕ ДЕЛИТ МЕХАНИЗМ С ПРОВЕРЯЕМЫМ. Числа берутся не из того, что вернул
поиск, а заново из PDF-корпуса на диске — то есть из источника, а не из
посредника. Если бы сверка шла с выдачей поиска, она подтвердила бы согласие
модели с поиском и промолчала бы о том, что оба ошиблись вместе.

Выход: 0 — все числа найдены на названных страницах; 1 — есть непроверенные;
2 — проверка не отработала (нет корпуса, нет ответов). ★Второе не равно
первому: «не смогли проверить» обязано отличаться от «проверено и чисто».
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# «[Отчёт EIA STEO, июль 2026, с. 34–35]» — и вариант с одной страницей.
CITATION = re.compile(r"\[Отчёт\s+([^,\]]+),\s*([а-яё]+\s+\d{4}),\s*с\.\s*(\d+)(?:\s*[–—-]\s*(\d+))?\]")
# Числа, которые вообще имеет смысл сверять: с десятичной частью. Целые вроде
# «2026» или «95%» — это годы и доли, они стоят в тексте ответа, а не в таблице,
# и требовать их дословного присутствия значило бы плодить ложные тревоги.
NUMBER = re.compile(r"\d+\.\d+")


# Любая законная разметка источника. Расчёт и веб проверить по корпусу нельзя,
# но их НАЛИЧИЕ проверить можно и нужно: число без всякой ссылки — отдельный и
# худший отказ, чем число со ссылкой на не ту страницу.
ANY_MARK = re.compile(r"\[(?:Отчёт|Расчёт|Источник|Веб-источник)")

# Ответ агента внутри демо-файла. Ниже идут служебные разделы («Что происходило
# внутри», список фрагментов с косинусами) — там числа законно стоят без ссылок,
# и требовать их разметки значило бы плодить ложные тревоги.
ANSWER = re.compile(r"### Ответ агента\n(.*?)\n---\n", re.S)


def answer_of(text: str) -> str:
    """Раздел с ответом, если файл — демо; иначе весь текст."""
    found = ANSWER.search(text)
    return found.group(1) if found else text


def blocks(text: str) -> list[str]:
    """Абзацы ответа: строки до пустой строки, склеенные в один кусок.

    ★СВЕРКА ИДЁТ БЛОКАМИ, А НЕ СТРОКАМИ, и это не украшение. Перечисление в
    ответе выглядит так: три строки с числами, и ссылка только на последней —
    она относится ко всему перечислению. Построчная сверка проверяла бы лишь
    последнюю строку, а остальные молча пропускала: замерено на демо, покрытие
    падало с 44 сверенных чисел до 19. Проверка, которая тихо сужает охват,
    опаснее отсутствующей — она печатает «не подтверждено: 0» ровно тем же
    тоном, что и полная.
    """
    result: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip():
            current.append(line)
        elif current:
            result.append(" ".join(current))
            current = []
    if current:
        result.append(" ".join(current))
    return result


def normalise(text: str) -> str:
    """Убрать то, что конвертер PDF расставляет произвольно."""
    return re.sub(r"[\s ]+", " ", text)


def load_corpus() -> dict:
    """Страницы всех отчётов: (дата, номер страницы) -> текст страницы."""
    from neftegaz.rag.ingest import parse_filename, read_pdf_pages

    pages: dict = {}
    for pdf in sorted((ROOT / "data" / "reports").glob("*.pdf")):
        meta = parse_filename(pdf.name)
        for page in read_pdf_pages(str(pdf)):
            pages[(meta.date, page["page"])] = normalise(page["text"])
    return pages


def main() -> int:
    # Каталог ответов задаётся первым доводом: это позволяет проверить саму
    # проверку на подложных ответах, не трогая настоящие.
    where = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "demo"
    answers = sorted(where.glob("*.md"))
    if not answers:
        print(f"проверка не отработала: в {where} нет ответов", file=sys.stderr)
        return 2
    pages = load_corpus()
    if not pages:
        print("проверка не отработала: корпус в data/reports пуст", file=sys.stderr)
        return 2
    print(f"корпус: {len(pages)} страниц, ответов: {len(answers)}")

    checked = missing = unmarked = 0
    for answer in answers:
        for line in blocks(answer_of(answer.read_text(encoding="utf-8"))):
            # ★Сначала — есть ли ссылка вообще. Абзац с числами и без всякой
            # разметки читается как собственное мнение системы; проверить его
            # читатель не может, а проверяемость и есть предмет поставки. Код в
            # блоках ``` — это дословный вывод инструмента, показанный как есть,
            # и разметки не требует.
            digits = NUMBER.findall(line)
            if digits and not ANY_MARK.search(line) and "```" not in line:
                unmarked += len(digits)
                print(f"  ✗ {answer.name}: числа {digits} стоят без ссылки: "
                      f"«{re.sub(r'[\\s ]+', ' ', line)[:70]}…»")
            # ★ССЫЛОК В СТРОКЕ МОЖЕТ БЫТЬ НЕСКОЛЬКО, и число засчитывается, если
            # оно стоит хоть на одной из названных страниц. Первая редакция
            # брала только первую ссылку и объявила 15 ложных тревог на строке
            # вида «в июле … , а в апреле … [Отчёт июль], [Отчёт апрель]»:
            # апрельские числа сверялись с июльскими страницами. Я едва не
            # доложил дефект проверки как дефект продукта.
            citations = list(CITATION.finditer(line))
            if not citations:
                continue
            haystack = ""
            named = []
            for citation in citations:
                _source, date, first, last = citation.groups()
                named.append(f"{date} с.{first}" + (f"-{last}" if last else ""))
                for number in range(int(first), int(last or first) + 1):
                    haystack += " " + pages.get((date, number), "")
            if not haystack.strip():
                print(f"  ✗ {answer.name}: страниц {', '.join(named)} в корпусе нет")
                missing += 1
                continue
            for number in NUMBER.findall(line):
                checked += 1
                if number not in haystack:
                    missing += 1
                    print(f"  ✗ {answer.name}: {number} нет ни на одной из {', '.join(named)}")

    print(f"\nсверено чисел: {checked}, не подтверждено: {missing}, "
          f"без ссылки: {unmarked}")
    if missing or unmarked:
        return 1
    # ★Ноль сверенных — не успех. Пустая проверка выдала бы «всё чисто» ровно
    # так же, как исправная, и это тот самый случай, когда проверка отвечает
    # одинаково при полном отказе проверяемого.
    if checked == 0:
        print("проверка не отработала: в ответах не нашлось ни одного числа со ссылкой",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
