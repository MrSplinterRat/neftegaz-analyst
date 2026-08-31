#!/usr/bin/env python3
"""Замер: какой поисковый сервис реально отвечает на вопросы нашего рода.

Зачем скрипт лежит в репозитории. Выбор сервиса в `WEB_BACKEND` — не вопрос
репутации, а вопрос факта: часть сервисов, к которым обращается библиотека
`ddgs`, не возвращает ничего, и понять это можно только прогоном. Числа в
документации стареют вместе с версией библиотеки, поэтому здесь лежит не
результат, а способ его получить заново.

    python scripts/probe_web_backends.py            # все текстовые сервисы
    python scripts/probe_web_backends.py brave yahoo

★Скрипт ходит в сеть по-настоящему: он для того и нужен. Каждый сервис
опрашивается ОТДЕЛЬНО (backend задан явно), поэтому видно поведение каждого,
а не то, кто успел ответить первым в режиме `auto`.
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

from neftegaz.tools.web import PREFERRED_DOMAINS, _domain, _is_denied  # noqa: E402

# Вопросы нашего рода, а не общие: сервис, хорошо ищущий рецепты, может ничего
# не знать про отраслевые публикации, и наоборот.
QUERIES = (
    "EIA short term energy outlook crude oil production forecast",
    "прогноз цены Brent на 2027 год",
    "OPEC+ production cut impact on oil prices",
)


def _is_preferred(domain: str) -> bool:
    return any(domain == good or domain.endswith(f".{good}") for good in PREFERRED_DOMAINS)


def main(argv: list[str]) -> int:
    try:
        from ddgs import DDGS
        from ddgs.engines import ENGINES
    except ImportError as exc:
        print(f"Библиотека веб-поиска не установлена: {exc}", file=sys.stderr)
        return 1

    backends = argv[1:] or sorted(ENGINES["text"])
    print(f"Вопросов: {len(QUERIES)}, сервисов: {len(backends)}")
    print(f"{'сервис':14s}{'результатов':>12s}{'предпочит.':>12s}{'отказов':>9s}{'секунд':>8s}")
    for backend in backends:
        total = preferred = denied = failures = 0
        started = time.monotonic()
        for query in QUERIES:
            try:
                with DDGS() as engine:
                    raw = list(engine.text(query, region="ru-ru", max_results=15, backend=backend))
            except Exception as exc:  # noqa: BLE001 - отказ сервиса и есть измеряемое
                failures += 1
                print(f"  ! {backend}: {type(exc).__name__}: {exc}"[:100])
                continue
            for item in raw:
                domain = _domain(item.get("href") or item.get("url") or "")
                if not domain:
                    continue
                total += 1
                if _is_denied(domain):
                    denied += 1
                elif _is_preferred(domain):
                    preferred += 1
        elapsed = time.monotonic() - started
        print(f"{backend:14s}{total:12d}{preferred:12d}{failures:9d}{elapsed:8.1f}")
        if denied:
            print(f"  (из них в чёрном списке: {denied})")

    print()
    print(
        "★Смотреть на ПОВТОРЯЕМОСТЬ, а не на разовый максимум: прогоните дважды "
        "с промежутком. Сервис, отвечающий через раз, в режиме auto выглядит "
        "работающим — просто вместо него отвечает кто-то другой."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
