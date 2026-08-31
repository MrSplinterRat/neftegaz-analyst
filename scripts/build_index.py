#!/usr/bin/env python3
"""Build the vector index over the PDF reports in data/reports/.

Usage:
    python scripts/build_index.py             # add new documents
    python scripts/build_index.py --recreate  # wipe and rebuild from scratch
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, ".")

from neftegaz.config import settings  # noqa: E402
from neftegaz.rag.index_stamp import write_stamp  # noqa: E402
from neftegaz.rag.ingest import ingest_directory  # noqa: E402
from neftegaz.rag.store import get_store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=settings.reports_dir)
    parser.add_argument("--recreate", action="store_true", help="удалить и построить заново")
    args = parser.parse_args()

    print(f"Каталог отчётов: {args.dir}")
    print(f"Модель эмбеддингов: {settings.embedding_model}")
    print(f"Хранилище: {settings.qdrant_url or settings.qdrant_path}")
    print()

    pdfs = sorted(name for name in os.listdir(args.dir) if name.lower().endswith(".pdf")) \
        if os.path.isdir(args.dir) else []
    if pdfs:
        print(f"PDF-файлов: {len(pdfs)}")
        print(
            "★Это долго: почти всё время уходит на кодировщик эмбеддингов, а не на разбор.\n"
            "  Замер на этой машине (один отчёт EIA STEO, 1177 фрагментов, 155.6 с):\n"
            "  разбор PDF ~0.5 с, нарезка ~0.02 с, эмбеддинг ~7.6 фрагмента/с —\n"
            "  то есть ~2.6 минуты на отчёт и ~21 минута на все восемь.\n"
            "  Ниже раз в 30 секунд печатается отметка «жив»: молчание в этом месте\n"
            "  неотличимо от зависания, а зависания здесь нет."
        )
        print(f"  Ожидаемое время: ~{len(pdfs) * 2.6:.0f} мин.")
        print(flush=True)

    started = time.monotonic()
    alive = threading.Event()

    def heartbeat() -> None:
        """Отметка «процесс жив» раз в 30 секунд.

        ★Только ВРЕМЯ, и это осознанно. Показывать долю сделанного было бы
        полезнее, но единственный способ узнать её — спрашивать хранилище во
        время записи, то есть вмешиваться в то, за чем наблюдаешь. Прибор,
        способный испортить наблюдаемое, здесь не стои́т своей точности.
        """
        while not alive.wait(30):
            print(f"  … идёт индексация, {(time.monotonic() - started) / 60:.1f} мин", flush=True)

    ticker = threading.Thread(target=heartbeat, daemon=True)
    ticker.start()
    try:
        results = ingest_directory(args.dir, recreate=args.recreate)
    finally:
        alive.set()
    elapsed = time.monotonic() - started

    if not results:
        print("PDF-файлов не найдено.")
        print(f"Положите отчёты в {args.dir} или запустите scripts/fetch_reports.py")
        return 1

    total = 0
    for name, count in sorted(results.items()):
        if count < 0:
            print(f"  ОШИБКА  {name}")
        else:
            print(f"  {count:5d} чанков  {name}")
            total += count

    print()
    print(f"Всего {total} чанков за {elapsed:.1f} с")
    in_store = get_store().count()
    print(f"В коллекции сейчас: {in_store} точек")
    # ★Правила сборки записываются РЯДОМ С ИНДЕКСОМ. Модель эмбеддингов, размер
    # фрагмента и перекрытие описывают сам индекс, а не поведение: поменяв их
    # без пересборки, вы получите систему, которая уверенно отвечает из индекса,
    # собранного по другим правилам. Отметка позволяет интерфейсу это заметить
    # и сказать вслух.
    stamp = write_stamp(in_store)
    print(f"Правила сборки записаны: {stamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
