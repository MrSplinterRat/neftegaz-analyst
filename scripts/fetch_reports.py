#!/usr/bin/env python3
"""Download a corpus of industry reports into data/reports/.

Source: the U.S. EIA Short-Term Energy Outlook — a monthly analytical report on
oil and gas supply, demand, prices and inventories, published openly and
without registration. The assignment names EIA alongside OPEC and the IEA; EIA
is the one of the three whose PDFs are directly downloadable, which is what
lets this script work on a fresh clone.

OPEC MOMR and IEA OMR are not fetched: OPEC serves its monthly report behind a
form (HTTP 403 to a direct request) and the IEA's is a paid product. To add
them, download manually and drop the files into data/reports/ following the
naming convention below — they will be indexed like any other document.

Naming convention (this is how citations get their name and date):
    EIA_STEO_2025-07.pdf  ->  [Отчёт EIA STEO, июль 2025, с. N]

Usage:
    python scripts/fetch_reports.py [--months 6] [--out data/reports]
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from datetime import date

CURRENT_URL = "https://www.eia.gov/outlooks/steo/pdf/steo_full.pdf"
ARCHIVE_URL = "https://www.eia.gov/outlooks/steo/archives/{stamp}.pdf"
TIMEOUT_SECONDS = 120
MIN_PDF_BYTES = 100_000  # a real STEO is several MB; anything tiny is an error page

MONTH_ABBR = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def previous_months(count: int) -> list[tuple[int, int]]:
    """The last ``count`` complete months, newest first, as (year, month)."""
    today = date.today()
    year, month = today.year, today.month
    result = []
    for _ in range(count):
        month -= 1
        if month == 0:
            month = 12
            year -= 1
        result.append((year, month))
    return result


def download(url: str, destination: str) -> int:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (neftegaz-analyst)"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        payload = response.read()

    # Check the magic bytes, not the status code: a server that answers 200
    # with an HTML "not found" page would otherwise leave us with a .pdf file
    # that is not a PDF, and the failure would surface much later, in the
    # indexer, as an unreadable document.
    if not payload.startswith(b"%PDF"):
        raise RuntimeError("ответ не является PDF")
    if len(payload) < MIN_PDF_BYTES:
        raise RuntimeError(f"слишком маленький файл ({len(payload)} байт)")

    with open(destination, "wb") as handle:
        handle.write(payload)
    return len(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", type=int, default=6, help="сколько архивных выпусков забрать")
    parser.add_argument("--out", default="data/reports")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    downloaded = 0

    targets: list[tuple[str, str]] = []
    for year, month in previous_months(args.months):
        stamp = f"{MONTH_ABBR[month - 1]}{str(year)[2:]}"
        targets.append((ARCHIVE_URL.format(stamp=stamp), f"EIA_STEO_{year}-{month:02d}.pdf"))

    for url, filename in targets:
        destination = os.path.join(args.out, filename)
        if os.path.exists(destination):
            print(f"  уже есть  {filename}")
            downloaded += 1
            continue
        try:
            size = download(url, destination)
            print(f"  {size / 1_048_576:5.1f} МБ  {filename}")
            downloaded += 1
        except Exception as exc:  # noqa: BLE001 - a missing month is not fatal
            print(f"  пропуск   {filename}: {exc}")

    print()
    if downloaded == 0:
        print("Ничего не скачано.", file=sys.stderr)
        print(
            "Положите PDF-отчёты в каталог вручную, соблюдая имя вида "
            "ИМЯ_ОТЧЁТА_ГГГГ-ММ.pdf",
            file=sys.stderr,
        )
        return 1

    print(f"Готово: {downloaded} отчётов в {args.out}")
    print("Дальше: python scripts/build_index.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
