"""Loading of historical oil price series.

The loader is deliberately defensive: real-world price exports are messy —
rows arrive unsorted, the same trading day may appear twice after a revision,
`close` may be blank on a holiday, and calendar days go missing on weekends.
Downstream forecasting assumes a *continuous daily* series, so the mess is
normalised here, once, instead of in every model.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["load_prices", "load_prices_from_frame"]


def load_prices(csv_path: str) -> pd.DataFrame:
    """Read a price CSV and return a continuous daily series.

    Expected input columns: ``date`` and ``close``.

    Returns a frame with a ``DatetimeIndex`` named ``date`` and a single
    ``close`` column of dtype float64, covering every calendar day between the
    first and last observation, with no missing values.

    Duplicate dates keep the **last** row in file order: price exports append
    revisions after the original row, so the later line is the corrected one.
    """
    frame = pd.read_csv(csv_path, dtype={"date": str})
    return load_prices_from_frame(frame)


def load_prices_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Same normalisation as :func:`load_prices`, for an in-memory frame.

    Kept separate so callers that already hold a frame (an API response, a
    test fixture) do not have to round-trip through a file.
    """
    missing = {"date", "close"} - set(frame.columns)
    if missing:
        raise KeyError(f"price frame is missing columns: {sorted(missing)}")

    frame = frame.copy()
    # errors="coerce" turns blanks and junk into NaN rather than raising:
    # a single bad cell must not cost us the whole series.
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.drop_duplicates(subset="date", keep="last")
    frame = frame.set_index("date").sort_index()

    full_calendar = pd.date_range(frame.index.min(), frame.index.max(), freq="D", name="date")
    frame = frame.reindex(full_calendar)

    # Forward fill carries the last traded price across weekends and holidays,
    # which is the correct convention for a closing price. The backward fill
    # only ever touches leading gaps, where there is no earlier price to carry.
    frame["close"] = frame["close"].ffill().bfill().astype("float64")
    return frame[["close"]]
