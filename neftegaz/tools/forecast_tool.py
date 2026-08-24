"""The price-forecasting tool the agent calls.

Requirement 2.5: a separate callable the agent invokes for questions like
"forecast Brent for 3 months" or "estimate the price range if output is cut by
N mb/d". It loads history, runs a model, and returns a forecast with a
confidence interval and a short interpretation.

The scenario adjustment (a supply cut or increase) is applied as a documented,
deliberately simple elasticity — see :func:`apply_supply_scenario`. It is
better to state a crude assumption plainly than to hide a sophisticated one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from neftegaz.config import settings
from neftegaz.forecast.data import load_prices
from neftegaz.forecast.models import ForecastResult, forecast

__all__ = ["ForecastReport", "run_forecast", "apply_supply_scenario", "PRICE_ELASTICITY"]

# Percent change in price per 1 mb/d change in supply.
#
# Short-run demand for oil is famously inelastic: estimates of the price
# elasticity of demand cluster around -0.05 to -0.10 over months. Inverting
# that, a 1 mb/d shift against ~102 mb/d of global supply (about 1%) moves
# price on the order of 10-20%. We take the conservative end, 10%, and we
# state it here rather than burying it, because this single number carries the
# whole scenario branch and a reviewer must be able to disagree with it.
PRICE_ELASTICITY = 10.0

GLOBAL_SUPPLY_MB_D = 102.0


@dataclass(frozen=True)
class ForecastReport:
    """Everything the agent needs to answer a forecast question."""

    instrument: str
    horizon_days: int
    method: str
    last_price: float
    point: float
    lower: float
    upper: float
    interpretation: str
    scenario: str | None
    frame: pd.DataFrame

    def as_text(self) -> str:
        lines = [
            f"Инструмент: {self.instrument}",
            f"Последняя известная цена: {self.last_price:.2f} долл./барр.",
            f"Горизонт: {self.horizon_days} дн.",
            f"Метод: {self.method}",
            f"Прогноз на конец горизонта: {self.point:.2f} долл./барр.",
            f"95% доверительный интервал: {self.lower:.2f} — {self.upper:.2f} долл./барр.",
        ]
        if self.scenario:
            lines.append(f"Сценарий: {self.scenario}")
        lines.append("")
        lines.append(self.interpretation)
        return "\n".join(lines)


def apply_supply_scenario(result: ForecastResult, supply_change_mb_d: float) -> ForecastResult:
    """Shift a forecast to reflect a change in supply.

    Positive ``supply_change_mb_d`` means more oil on the market and so a lower
    price. The whole band shifts by the same proportion: the scenario changes
    the level, not the uncertainty about the level.
    """
    if supply_change_mb_d == 0:
        return result
    share = supply_change_mb_d / GLOBAL_SUPPLY_MB_D
    multiplier = 1.0 - share * PRICE_ELASTICITY
    # A scenario extreme enough to drive the multiplier non-positive is outside
    # what a linear elasticity can express; refuse rather than print a negative
    # oil price.
    if multiplier <= 0:
        raise ValueError(
            f"supply change of {supply_change_mb_d} mb/d is outside the range this "
            "linear elasticity model can represent"
        )
    shifted = result.frame * multiplier
    direction = "сокращение" if supply_change_mb_d < 0 else "рост"
    return ForecastResult(
        frame=shifted,
        method=f"{result.method} + сценарий предложения",
        params={**result.params, "supply_change_mb_d": supply_change_mb_d},
        residual_sigma=result.residual_sigma * multiplier,
        n_observations=result.n_observations,
    )


def run_forecast(
    horizon_days: int = 90,
    method: str = "auto",
    supply_change_mb_d: float = 0.0,
    instrument: str = "Brent",
    prices_csv: str | None = None,
) -> ForecastReport:
    """Load history, forecast, optionally apply a supply scenario.

    Raises FileNotFoundError with an actionable message when the price file is
    absent — that is a setup problem, and silently returning a made-up series
    would be the worst possible response for this particular product.
    """
    path = prices_csv or settings.prices_csv
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"price history not found at {path}. Run scripts/fetch_prices.py to "
            "download it, or set PRICES_CSV in .env to point at your own file."
        )

    history = load_prices(path)
    series = history["close"]
    result = forecast(series, horizon_days, method=method)

    scenario_text = None
    if supply_change_mb_d:
        result = apply_supply_scenario(result, supply_change_mb_d)
        direction = "сокращение" if supply_change_mb_d < 0 else "увеличение"
        scenario_text = (
            f"{direction} предложения на {abs(supply_change_mb_d):.2f} млн барр./сут "
            f"(эластичность {PRICE_ELASTICITY:.0f}% на 1 млн барр./сут)"
        )

    last_row = result.frame.iloc[-1]
    return ForecastReport(
        instrument=instrument,
        horizon_days=horizon_days,
        method=result.method,
        last_price=float(series.iloc[-1]),
        point=float(last_row["forecast"]),
        lower=float(last_row["lower"]),
        upper=float(last_row["upper"]),
        interpretation=result.interpretation(instrument),
        scenario=scenario_text,
        frame=result.frame,
    )
