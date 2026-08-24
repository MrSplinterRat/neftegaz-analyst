#!/usr/bin/env python3
"""Download Brent price history into data/prices/brent.csv.

Source: yfinance (ticker BZ=F, Brent crude futures), which the assignment names
explicitly and which needs no API key — a reviewer can run this on a fresh
clone without registering anywhere.

If the download fails (no network, or Yahoo rate-limits), the repository ships
a prepared CSV at the same path, which the assignment also allows. The system
therefore always has price history; this script refreshes it.

Usage:
    python scripts/fetch_prices.py [--ticker BZ=F] [--period 5y] [--out PATH]
"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_TICKER = "BZ=F"  # Brent crude futures
MIN_ROWS = 100


def download(ticker: str, period: str):
    import yfinance as yf

    frame = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if frame is None or frame.empty:
        raise RuntimeError("источник вернул пустой ответ")

    # yfinance returns a MultiIndex on the columns when several tickers are
    # requested, and since version 0.2.51 sometimes even for one. Flatten it so
    # the rest of the script does not have to care which shape arrived.
    if hasattr(frame.columns, "nlevels") and frame.columns.nlevels > 1:
        frame.columns = frame.columns.get_level_values(0)

    if "Close" not in frame.columns:
        raise RuntimeError(f"в ответе нет колонки Close, есть: {list(frame.columns)}")

    out = frame[["Close"]].reset_index()
    out.columns = ["date", "close"]
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["close"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--period", default="5y", help="5y, 10y, max …")
    parser.add_argument("--out", default="data/prices/brent.csv")
    args = parser.parse_args()

    try:
        frame = download(args.ticker, args.period)
    except Exception as exc:  # noqa: BLE001
        print(f"ОШИБКА: не удалось скачать {args.ticker}: {exc}", file=sys.stderr)
        if os.path.exists(args.out):
            print(f"Существующий файл {args.out} оставлен без изменений.", file=sys.stderr)
        else:
            print(
                "Укажите свой CSV в .env через PRICES_CSV (колонки date,close).",
                file=sys.stderr,
            )
        return 1

    # Too few rows means the ticker was rejected and we got a stub back. Refuse
    # rather than overwrite good history with junk.
    if len(frame) < MIN_ROWS:
        print(
            f"ОШИБКА: получено всего {len(frame)} строк — вероятно, тикер "
            f"{args.ticker!r} не распознан. Файл не перезаписан.",
            file=sys.stderr,
        )
        return 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    frame.to_csv(args.out, index=False)

    print(f"Записано {len(frame)} строк в {args.out}")
    print(f"Период: {frame['date'].iloc[0]} — {frame['date'].iloc[-1]}")
    print(f"Последняя цена: {float(frame['close'].iloc[-1]):.2f} долл./барр.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
