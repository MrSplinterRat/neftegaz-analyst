#!/usr/bin/env python3
"""Стенд приёмки разборщиков PDF: кто восстанавливает таблицу, а кто её портит.

ЗАЧЕМ. Сейчас таблицы восстанавливаются регулярками поверх плоского текста, и
структура ВЫВОДИТСЯ из следов вёрстки. Отсюда три дефекта за один день: у 903
строк из 938 отрезался последний столбец, шапка не находилась из-за склейки с
предыдущей фразой, а вместо шапки бралась годовая полоса. Кандидат на замену
обязан не рассказывать про поддержку таблиц, а показать числа.

★ГЛАВНАЯ МЕРА — НЕ «СКОЛЬКО ТАБЛИЦ НАШЛОСЬ», А СКОЛЬКО ЧИСЕЛ ВСТАЛО НЕ В СВОЮ
КОЛОНКУ. Пропуск виден в ответе и вызывает вопрос; сдвиг на один столбец
читается как обычный ответ и врёт числом. Поэтому стенд отдельно ищет сдвиг:
если ряд совпадает со смещением, это докладывается как сдвиг, а не как
пятнадцать разных ошибок.

★ЭТАЛОН СНЯТ С ОТРИСОВАННЫХ СТРАНИЦ, а не с вывода pypdf. Эталон, снятый
проверяемым механизмом, награждал бы кандидата за воспроизведение его ошибок.

Запуск:
    python3 bench.py                 # все адаптеры
    python3 bench.py pdfplumber      # только названные
    python3 bench.py --selftest      # проверить сам стенд (см. ниже)
    python3 bench.py --json out.json # машиночитаемый отчёт

★САМОПРОВЕРКА ОБЯЗАТЕЛЬНА. Стенд, который умеет только соглашаться,
неотличим от отсутствия стенда. Поэтому рядом с настоящими адаптерами лежат
вредители: один отрезает последний столбец, другой сдвигает ряд на единицу,
третий выбрасывает прочерки. `--selftest` требует, чтобы КАЖДЫЙ из них был
пойман, и возвращает 1, если хоть один прошёл как исправный.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
MAX_SHIFT = 3


# ── сверка ─────────────────────────────────────────────────────────────────


def norm(value: str) -> str:
    """Приведение ячейки к сравнимому виду: пробелы и запятые разрядов не значат.

    Прочерк НЕ приравнивается к пустой строке. Разница между «данных нет» и
    «ячейки нет» — это ровно та разница, из-за которой ряд уезжает на столбец.
    """
    return str(value).strip().replace(",", "").replace("−", "-").replace("–", "-")


def find_shift(actual: list[str], expected: list[str]) -> tuple[int, int]:
    """Смещение, при котором совпадений больше всего, и число несовпадений.

    Возвращает (сдвиг, ошибок при этом сдвиге). Сдвиг 0 означает, что ряд стоит
    на месте; ненулевой — что значения есть, но не в своих колонках.
    """
    best = (0, len(expected) + 1)
    for shift in range(-MAX_SHIFT, MAX_SHIFT + 1):
        wrong = 0
        for index, want in enumerate(expected):
            source = index + shift
            got = actual[source] if 0 <= source < len(actual) else None
            if got is None or norm(got) != norm(want):
                wrong += 1
        if wrong < best[1]:
            best = (shift, wrong)
    return best


def match(candidates: list[str], needle: str) -> int:
    """Индекс первого элемента, содержащего needle (без учёта регистра); -1 иначе."""
    lowered = needle.lower()
    for index, item in enumerate(candidates):
        if lowered in str(item).lower():
            return index
    return -1


def score_columns(actual: list[str], table: dict) -> str:
    """Насколько кандидат восстановил шапку. Три исхода, и они не равноценны.

    «двухъярусная» — вернул привязку квартала к году (2025Q1 …). Это то, чего
    из плоского текста получить нельзя в принципе, и ради чего всё затевалось.
    «нижний ярус» — вернул только кварталы: лучше, чем ничего, но «какой это
    Q1» по-прежнему неизвестно. «нет» — шапки не отдал.
    """
    if not actual:
        return "нет"
    plain = [norm(c) for c in actual]
    if plain == [norm(c) for c in table["колонки"]]:
        return "двухъярусная"
    if plain == [norm(c) for c in table["колонки_нижний_ярус"]]:
        return "нижний ярус"
    return f"своя ({len(actual)} стлб)"


def score(parsed: list[dict], truth: dict) -> dict:
    """Свести разбор кандидата с эталоном. Возвращает отчёт одним словарём."""
    captions = [t.get("caption", "") for t in parsed]
    report = {
        "таблиц_найдено": 0,
        "таблиц_ожидалось": len(truth["таблицы"]),
        "строк_найдено": 0,
        "строк_ожидалось": sum(len(t["строки"]) for t in truth["таблицы"]),
        "ячеек_ожидалось": sum(len(r["значения"]) for t in truth["таблицы"] for r in t["строки"]),
        "не_в_своей_колонке": 0,
        "сдвиги": [],
        "шапка": [],
        "подробности": [],
    }

    for table in truth["таблицы"]:
        index = match(captions, table["заголовок"][:40])
        if index < 0:
            report["подробности"].append(f"таблица не найдена: {table['заголовок'][:50]}")
            # Ненайденная таблица — это все её ячейки мимо, и молчать об этом
            # нельзя: иначе кандидат, не нашедший ничего, покажет ноль ошибок.
            report["не_в_своей_колонке"] += sum(len(r["значения"]) for r in table["строки"])
            continue
        report["таблиц_найдено"] += 1
        found = parsed[index]
        report["шапка"].append(score_columns(found.get("columns") or [], table))

        labels = [r.get("label", "") for r in found.get("rows") or []]
        for row in table["строки"]:
            position = match(labels, row["подпись"])
            if position < 0:
                report["подробности"].append(f"строка не найдена: {row['подпись']}")
                report["не_в_своей_колонке"] += len(row["значения"])
                continue
            report["строк_найдено"] += 1
            actual = [str(v) for v in found["rows"][position].get("values") or []]
            shift, wrong = find_shift(actual, row["значения"])
            report["не_в_своей_колонке"] += wrong
            if shift:
                report["сдвиги"].append(f"{row['подпись']}: сдвиг {shift:+d}")
            if wrong:
                report["подробности"].append(
                    f"{row['подпись']}: ошибок {wrong}/{len(row['значения'])}"
                    f"{f', сдвиг {shift:+d}' if shift else ''}"
                    f" | ожидалось {row['значения'][:4]}… получено {actual[:4]}…"
                )
    return report


# ── адаптеры ───────────────────────────────────────────────────────────────


def adapters(only: list[str]) -> list[tuple[str, object]]:
    """Загрузить адаптеры из каталога adapters/. Служебные (_*) пропускаются."""
    found = []
    for path in sorted((ROOT / "adapters").glob("*.py")):
        if path.stem.startswith("_"):
            continue
        if only and path.stem not in only:
            continue
        try:
            module = importlib.import_module(f"adapters.{path.stem}")
        except Exception as exc:  # noqa: BLE001 — не установлен = законный исход
            found.append((path.stem, path.stem, exc))
            continue
        found.append((getattr(module, "NAME", path.stem), path.stem, module))
    return found


def run(only: list[str], truth: dict) -> list[dict]:
    document = str(PROJECT / truth["документ"])
    results = []
    for name, stem, module in adapters(only):
        if isinstance(module, Exception):
            results.append(
                {"кандидат": name, "_модуль": stem, "ошибка": f"не загрузился: {module}"}
            )
            continue
        started = time.time()
        try:
            parsed = module.parse(document)
        except Exception as exc:  # noqa: BLE001
            results.append({"кандидат": name, "_модуль": stem, "ошибка": f"упал на разборе: {exc}"})
            continue
        elapsed = time.time() - started
        report = score(parsed, truth)
        report["кандидат"] = name
        report["_модуль"] = stem
        report["секунд"] = round(elapsed, 2)
        report["вредитель"] = bool(getattr(module, "SABOTEUR", False))
        results.append(report)
    return results


# ── вывод ──────────────────────────────────────────────────────────────────


def render(results: list[dict]) -> None:
    # ★Сдвиг вынесен ОТДЕЛЬНЫМ столбцом, а не спрятан в число ошибок. Кандидат,
    # съехавший на колонку, показывает мало несовпадений (крайние значения просто
    # выпадают за край) и в сводке выглядит почти чистым. Именно он опаснее всех:
    # все числа настоящие, ответ правдоподобен, и врёт он молча.
    head = f"{'кандидат':<24}{'таблиц':>7}{'строк':>7}{'НЕ В СВОЕЙ':>12}{'СДВИГ':>7}{'шапка':>15}{'сек':>7}"
    print(head)
    print("─" * len(head))
    for row in results:
        if "ошибка" in row:
            print(f"{row['кандидат']:<22}{row['ошибка']}")
            continue
        header = "/".join(sorted(set(row["шапка"]))) or "нет"
        shifts = "ЕСТЬ" if row["сдвиги"] else "—"
        print(
            f"{row['кандидат']:<24}"
            f"{row['таблиц_найдено']}/{row['таблиц_ожидалось']:<5}"
            f"{row['строк_найдено']}/{row['строк_ожидалось']:<5}"
            f"{row['не_в_своей_колонке']:>7}/{row['ячеек_ожидалось']:<4}"
            f"{shifts:>7}{header:>15}{row['секунд']:>7}"
        )
    print()
    for row in results:
        if row.get("подробности"):
            print(f"── {row['кандидат']} ──")
            for line in row["подробности"]:
                print(f"   {line}")


def selftest(results: list[dict]) -> int:
    """Приёмка самого стенда: каждый вредитель ОБЯЗАН быть пойман, а идеал — пройти.

    Зелёный вредитель означает, что зелёная строка настоящего кандидата ничего
    не значит. Красный идеал означает обратную беду: стенд, который всегда
    находит ошибку, так же бесполезен, как стенд, который всегда молчит.

    ★ОЖИДАЕМЫЙ СОСТАВ КОНТРОЛЕЙ БЕРЁТСЯ С ДИСКА, А НЕ ИЗ РЕЗУЛЬТАТОВ. Первая
    редакция считала вредителей по полю в отчёте — и когда все четыре контроля
    не загрузились из-за опечатки в импорте, самопроверка написала «пройдена,
    пойманы все вредители (0)». Проверка делила механизм с проверяемым:
    сломался загрузчик — и контроль исчез вместе с ним, не подав признака.
    """
    expected = {path.stem for path in (ROOT / "adapters").glob("saboteur_*.py")}
    reference = {path.stem for path in (ROOT / "adapters").glob("reference_*.py")}
    if not expected:
        print("\nСАМОПРОВЕРКА ПРОВАЛЕНА: вредителей на диске нет вовсе.")
        return 1

    by_name = {r.get("_модуль", r["кандидат"]): r for r in results}
    broken = []
    for stem in sorted(expected | reference):
        row = by_name.get(stem)
        if row is None:
            broken.append(f"{stem}: контроль не дошёл до стенда")
        elif "ошибка" in row:
            broken.append(f"{stem}: не отработал ({row['ошибка']})")
        elif stem in expected and row["не_в_своей_колонке"] == 0:
            broken.append(f"{stem}: ПРОШЁЛ как исправный — стенд слеп")
        elif stem in reference and row["не_в_своей_колонке"] != 0:
            broken.append(
                f"{stem}: идеальный разбор объявлен неверным "
                f"({row['не_в_своей_колонке']} ячеек) — стенд врёт в другую сторону"
            )

    print()
    if broken:
        print("САМОПРОВЕРКА ПРОВАЛЕНА:")
        for line in broken:
            print(f"   {line}")
        return 1
    print(
        f"Самопроверка пройдена: пойманы все вредители ({len(expected)}), "
        f"идеал признан чистым ({len(reference)})."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapters", nargs="*", help="какие адаптеры гонять (по умолчанию все)")
    parser.add_argument("--selftest", action="store_true", help="проверить сам стенд вредителями")
    parser.add_argument("--json", metavar="ФАЙЛ", help="машиночитаемый отчёт")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    truth = json.loads((ROOT / "truth.json").read_text(encoding="utf-8"))
    results = run(args.adapters, truth)
    render(results)
    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return selftest(results) if args.selftest else 0


if __name__ == "__main__":
    raise SystemExit(main())
