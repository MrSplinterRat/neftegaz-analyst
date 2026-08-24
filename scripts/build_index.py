#!/usr/bin/env python3
"""Build the vector index over the PDF reports in data/reports/.

Usage:
    python scripts/build_index.py             # add new documents
    python scripts/build_index.py --recreate  # wipe and rebuild from scratch
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, ".")

from neftegaz.config import settings  # noqa: E402
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

    started = time.monotonic()
    results = ingest_directory(args.dir, recreate=args.recreate)
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
    print(f"В коллекции сейчас: {get_store().count()} точек")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
