#!/usr/bin/env python3
"""Прогон списка вопросов через агента с сохранением стенограмм.

Отличие от run_demo.py: тот показывает пять обязательных сценариев ТЗ, этот —
рабочий инструмент отладки. Вопросы подобраны так, чтобы бить в места, где
разбор PDF решает: значение конкретной ячейки за конкретный квартал, сравнение
кварталов внутри года, годовой итог против квартала. До перехода на
координатный разбор такие вопросы получали строку из соседней таблицы.

Usage: python scripts/ask_batch.py [--out /path/dir] [--only 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neftegaz.agent.graph import answer_question  # noqa: E402

QUESTIONS = [
    "Какой прогноз EIA по цене Brent на четвёртый квартал 2026 года?",
    "Сравни прогноз EIA по WTI на кварталы 2026 года — где пик и где спад?",
    "Какая среднегодовая цена Brent прогнозируется на 2027 год?",
    "Сколько нефти добывают страны ОПЕК, участвующие в соглашении ОПЕК+?",
    "Что EIA прогнозирует по мировому потреблению жидких топлив в 2027 году?",
    "Какие цены на пропан для населения указаны в последнем отчёте?",
    "Спрогнозируй цену Brent на 90 дней вперёд.",
    "Что происходило с ценами на нефть в последние недели по новостям?",
    "Сопоставь прогноз EIA по Brent с текущими рыночными котировками.",
    "Посоветуй, какие акции купить прямо сейчас.",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/home/claude/tasks/neftegaz-qa")
    parser.add_argument("--only", type=int, help="только вопрос с этим номером")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for number, question in enumerate(QUESTIONS, start=1):
        if args.only and number != args.only:
            continue
        print(f"\n{'=' * 70}\n[{number}] {question}", flush=True)
        started = time.time()
        try:
            result = answer_question(question)
        except Exception:  # noqa: BLE001 — один упавший вопрос не должен рвать прогон
            print(traceback.format_exc(), flush=True)
            result = {"error": traceback.format_exc()}
        spent = time.time() - started

        answer = result.get("answer", "") if isinstance(result, dict) else str(result)
        route = result.get("route", "") if isinstance(result, dict) else ""
        report_hits = result.get("report_hits", []) if isinstance(result, dict) else []
        web_hits = result.get("web_hits", []) if isinstance(result, dict) else []

        print(
            f"--- маршрут: {route} | {spent:.1f} с | из отчётов: {len(report_hits)}"
            f" | из сети: {len(web_hits)}",
            flush=True,
        )
        print(answer, flush=True)
        for hit in report_hits[:6]:
            page = getattr(hit, "page_start", "?")
            name = getattr(hit, "source_name", "?")
            date = getattr(hit, "doc_date", "")
            score = getattr(hit, "score", 0.0)
            context = (getattr(hit, "context", "") or "").split("\n")[0]
            print(f"    [{score:.2f}] {name} {date}, с. {page}  {context}", flush=True)

        (out / f"q{number:02d}.json").write_text(
            json.dumps(
                {"question": question, "seconds": round(spent, 1), "result": result},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
