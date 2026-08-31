#!/usr/bin/env python3
"""Проверка требования ТЗ 2.1: ни одного числа, которого нет в источнике.

Каждое число ответа сверяется с текстом фрагментов, РЕАЛЬНО поданных модели в
контекст, и с выходом расчётного модуля. Число, не найденное ни там, ни там, —
кандидат в выдуманные.

★ПОЧЕМУ ИМЕННО ПОДАННЫЕ, А НЕ НАЙДЕННЫЕ. Найдено может быть больше, чем влезло
в бюджет контекста. Сверять с найденным значило бы прощать числа из фрагментов,
которых модель не видела, — то есть проверять не то, что происходило.

★ПОЧЕМУ ПРОВЕРКА НЕ ДЕЛИТ МЕХАНИЗМ С ПРОВЕРЯЕМЫМ. Список поданного берётся не
пересчётом по своим правилам, а вызовом `fed_report_hits` — той самой функции,
которой пользуется узел ответа. Своя копия отбора отстала бы от боевой при
первой же правке бюджета и не подала бы об этом ни одного признака.

⚠ЧЕГО ЭТА ПРОВЕРКА НЕ ДЕЛАЕТ. Она не сверяет число с ЕГО СОБСТВЕННОЙ ссылкой:
число, взятое из одного фрагмента и приписанное другому, здесь пройдёт. Это
отдельная, более строгая мера, и она пока не сделана.

Запуск:
    python scripts/check_numbers.py                 # набор по умолчанию
    python scripts/check_numbers.py --self-check    # проверить сам инструмент
"""

from __future__ import annotations

import argparse
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neftegaz.agent.graph import answer_question, fed_report_hits  # noqa: E402

# Число: цифры с необязательными разделителями. ⚠ПРОБЕЛ В КЛАСС НЕ ВХОДИТ, и это
# оплаченная граблина. Первая версия писалась как `\d[\d ,.]*\d` — с пробелом
# внутри класса — и на строке таблицы «20.31 20.51 20.97» выдавала ОДНО число
# «2031205120.97» вместо трёх. Настоящие числа фрагмента при этом исчезали из
# списка разрешённых, а числа ответа объявлялись невзятыми: прогон показал
# 59 «выдуманных» чисел из 252, и ни одного из них не существовало. Случай
# закреплён в `self_check` ниже, чтобы граблина не вернулась молча.
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)*")

# Разделитель разрядов: группы ровно по три цифры («1,258», «13.600.000»).
_THOUSANDS = re.compile(r"^-?\d{1,3}(?:[.,]\d{3})+$")

# Ссылка на страницу — наша собственная разметка, а не утверждение о мире.
# Её номера в сверку не идут: страницу проставляет код по метаданным фрагмента.
_CITATION = re.compile(r"\[[^\]\n]*?с\.\s*\d+(?:\s*[–-]\s*\d+)?[^\]\n]*\]")

QUESTIONS = [
    "Какой прогноз EIA по добыче нефти в США на следующий год?",
    "Что говорят отчёты о запасах нефти в США?",
    "Спрогнозируй цену Brent на 3 месяца",
    "Оцени диапазон цен при сокращении добычи ОПЕК+ на 1.5 млн барр./сут",
    "Что с ценами на природный газ в Европе?",
    "Какой прогноз по мировому спросу на нефть?",
]


def normalise(raw: str) -> str:
    """Число к единому виду, чтобы «13,6», «13.6» и «13.60» были одним числом."""
    token = raw.strip()
    if _THOUSANDS.match(token):
        return token.replace(",", "").replace(".", "")
    token = token.replace(",", ".")
    parts = token.split(".")
    if len(parts) > 2:  # несколько разделителей — считаем последний дробным
        token = "".join(parts[:-1]) + "." + parts[-1]
    try:
        value = float(token)
    except ValueError:
        return token
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def numbers_in(text: str) -> list[str]:
    return [normalise(m.group()) for m in _NUMBER.finditer(text) if normalise(m.group())]


def allowed_numbers(state: dict, question: str) -> set[str]:
    """Всё, из чего ответ имеет право брать числа."""
    sources = [question, state.get("forecast_text", "") or ""]
    for hit in fed_report_hits(state.get("report_hits") or []):
        sources.append(hit.text)
        sources.append(getattr(hit, "context", "") or "")
        # Номера страниц фрагмента — законный источник для ссылки.
        sources.append(f"{hit.page} {hit.page_end}")
    for hit in state.get("web_hits") or []:
        sources.append(getattr(hit, "snippet", "") or "")
        sources.append(getattr(hit, "title", "") or "")
    allowed: set[str] = set()
    for source in sources:
        allowed.update(numbers_in(source))
    return allowed


def unsupported(answer: str, allowed: set[str]) -> list[str]:
    """Числа ответа, которых нет ни в одном разрешённом источнике."""
    body = _CITATION.sub(" ", answer)
    return [value for value in numbers_in(body) if value not in allowed]


def self_check() -> int:
    """★Проверка самого инструмента: он обязан ловить И пропускать.

    Инструмент, который ничего не находит, неотличим от исправной системы, а
    находящий всё — от сломанной. Обе стороны проверяются здесь, до того как
    числа пойдут в отчёт.
    """
    checks = []

    split = numbers_in("20.31 20.51 20.97")
    checks.append(("соседние числа таблицы не склеены", split == ["20.31", "20.51", "20.97"], split))

    same = numbers_in("13,6") == numbers_in("13.60") == numbers_in("13.6")
    checks.append(("13,6 и 13.60 — одно число", same, numbers_in("13,60")))

    thousand = numbers_in("1,258")
    checks.append(("1,258 — это 1258, а не 1.258", thousand == ["1258"], thousand))

    allowed = {"13.6", "70", "2027"}
    caught = unsupported("Добыча вырастет до 13,6 млн барр./сут к 2027 году.", allowed)
    checks.append(("подтверждённые числа не помечены", caught == [], caught))

    missed = unsupported("Добыча составит 99,9 млн барр./сут.", allowed)
    checks.append(("выдуманное число поймано", missed == ["99.9"], missed))

    cite = unsupported("Цена 70 долларов [Отчёт EIA STEO, июль 2026, с. 52].", allowed)
    checks.append(("номер страницы в ссылке не в счёт", cite == [], cite))

    for name, ok, got in checks:
        print(f"  [{'ок' if ok else 'ПРОВАЛ'}] {name}: {got}")
    good = all(ok for _, ok, _ in checks)
    print("самопроверка:", "ПРОЙДЕНА" if good else "ПРОВАЛЕНА")
    return 0 if good else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="проверить сам инструмент")
    args = parser.parse_args()
    if args.self_check:
        return self_check()

    total_numbers = 0
    total_unsupported = 0
    clean_answers = 0
    for question in QUESTIONS:
        state = answer_question(question, thread_id=f"numcheck-{uuid.uuid4().hex[:8]}")
        answer = state.get("answer", "")
        allowed = allowed_numbers(state, question)
        bad = unsupported(answer, allowed)
        count = len(numbers_in(_CITATION.sub(" ", answer)))
        total_numbers += count
        total_unsupported += len(bad)
        clean_answers += not bad
        mark = "чисто" if not bad else f"НЕ ПОДТВЕРЖДЕНО {len(bad)}"
        fed = len(fed_report_hits(state.get("report_hits") or []))
        print(f"[{mark}] {question}")
        print(f"    чисел в ответе {count}, поданных фрагментов {fed}, маршрут {state.get('route')}")
        if bad:
            print(f"    {', '.join(sorted(set(bad))[:12])}")

    print(
        f"\nИТОГ: ответов без единого неподтверждённого числа {clean_answers} из "
        f"{len(QUESTIONS)}; неподтверждённых чисел {total_unsupported} из {total_numbers}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
