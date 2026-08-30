"""Квартальные ряды спроса, предложения и запасов, вычитанные из корпуса STEO.

Зачем это здесь. Обе прежние модели прогноза работают только с историей цены:
они не знают ни сколько нефти добывают, ни сколько её потребляют, ни сколько
лежит в хранилищах. Значит, они не могут ответить на вопрос, ради которого
аналитику и нужен прогноз, — ПОЧЕМУ цена будет такой.

Ряды берутся не из внешнего API, а из ТОГО ЖЕ корпуса отчётов, по которому
система отвечает на вопросы. Это не обходной путь (``api.eia.gov`` без ключа
отвечает 403), а более правильное устройство: прогноз начинает опираться на
данные, вычитанные нашей же дугой восприятия, и любое число в нём можно
проследить до страницы отчёта.

Что берётся:

* ``Table 3a`` — мировая добыча, мировое потребление, запасы OECD на конец
  периода, изменение запасов;
* ``Table 2``  — средняя цена Brent за квартал.

★ВЫПУСК ИМЕЕТ ЗНАЧЕНИЕ. Каждый STEO печатает и прошлое, и прогноз, а прошлое
пересматривает: добыча за 2026Q1 в декабрьском выпуске — 106.50 млн барр./сут,
в июльском — 103.86. Разница в 2.6 млн барр./сут больше, чем шок, который мы
собираемся измерять. Поэтому у каждого наблюдения хранится выпуск, из которого
оно взято, и факт отделён от прогноза: квартал считается фактическим только в
тех выпусках, которые вышли после его окончания.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from neftegaz.config import settings

__all__ = [
    "FactorSeries",
    "load_factors",
    "load_all_vintages",
    "projection_error_sigma",
    "read_report_factors",
    "quarter_end",
    "vintage_of",
]

# Строки Table 3a, которые нам нужны, по их порядковому номеру внутри таблицы.
# ★Именно по номеру, а не по подписи: подписи в таблице ПОВТОРЯЮТСЯ — «World
# total» стоит и у добычи, и у потребления, и у изменения запасов. Разделяют их
# только заголовки секций, которые в разобранной таблице отдельными строками не
# лежат. Номера проверены на всех восьми выпусках корпуса: структура таблицы у
# них одинаковая (31 строка, 15 колонок).
ROW_PRODUCTION = 0  # World total production
ROW_CONSUMPTION = 10  # World total consumption
ROW_STOCK_CHANGE = 24  # World total inventory net withdrawals
ROW_OECD_INVENTORIES = 28  # OECD commercial inventories, end of period

QUARTER_COLUMN = re.compile(r"^(\d{4})Q([1-4])$")
VINTAGE_IN_NAME = re.compile(r"(\d{4})-(\d{2})")


@dataclass(frozen=True)
class FactorSeries:
    """Панель факторов плюс то, что нужно, чтобы ей верить.

    ``frame`` — по строке на квартал, с колонками ``production``,
    ``consumption``, ``stock_change``, ``oecd_inventories``, ``brent`` и
    служебными ``vintage`` (из какого выпуска взято) и ``actual`` (факт или
    прогноз).
    """

    frame: pd.DataFrame
    reports_read: int
    reports_failed: list[str]

    @property
    def actuals(self) -> pd.DataFrame:
        return self.frame[self.frame["actual"]].drop(columns=["actual"])

    @property
    def projections(self) -> pd.DataFrame:
        return self.frame[~self.frame["actual"]].drop(columns=["actual"])


def quarter_end(label: str) -> pd.Timestamp | None:
    """Последний день квартала, названного меткой колонки вида ``2026Q1``."""
    found = QUARTER_COLUMN.match(label.strip())
    if not found:
        return None
    year, quarter = int(found.group(1)), int(found.group(2))
    return pd.Period(f"{year}Q{quarter}", freq="Q").end_time.normalize()


def vintage_of(path: str) -> pd.Timestamp | None:
    """Месяц выпуска отчёта, взятый из имени файла (``EIA_STEO_2026-07.pdf``)."""
    found = VINTAGE_IN_NAME.search(os.path.basename(path))
    if not found:
        return None
    return pd.Timestamp(year=int(found.group(1)), month=int(found.group(2)), day=1)


def _number(cell: str | None) -> float | None:
    """Число из ячейки таблицы, с разделителем тысяч и знаком минус.

    Пустая ячейка и прочерк — это ОТСУТСТВИЕ значения, а не ноль: ноль здесь
    означал бы «добыча остановилась», и подстановка его вместо пропуска сдвинет
    любую оценку, построенную на этих рядах.
    """
    if cell is None:
        return None
    text = str(cell).strip().replace(",", "").replace("−", "-").replace("–", "-")
    if not text or text in {"-", "--", "NA", "NM"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _find_table(document, prefix: str):
    for page in document.pages:
        for table in page.tables:
            if table.caption.strip().startswith(prefix):
                return table
    return None


def read_report_factors(path: str) -> pd.DataFrame:
    """Вычитать квартальные ряды из одного отчёта.

    Возвращает пустой кадр, если нужных таблиц в отчёте нет — отсутствие
    таблицы это состояние документа, а не сбой программы, и пачка отчётов не
    должна разваливаться из-за одного нестандартного выпуска.
    """
    import pdf2xml

    document = pdf2xml.parse_pdf(path)
    balance = _find_table(document, "Table 3a")
    prices = _find_table(document, "Table 2.")
    if balance is None or prices is None:
        return pd.DataFrame()

    if len(balance.rows) <= ROW_OECD_INVENTORIES:
        return pd.DataFrame()

    brent_rows = [row for row in prices.rows if "brent" in row.label.lower()]
    brent = brent_rows[0] if brent_rows else None

    vintage = vintage_of(path)
    records = []
    for index, column in enumerate(balance.columns):
        end = quarter_end(column.label)
        if end is None:  # годовые колонки (2025, 2026, 2027) нам не нужны
            continue
        records.append(
            {
                "quarter": column.label,
                "quarter_end": end,
                "vintage": vintage,
                # Квартал фактический, если он закончился ДО выхода выпуска.
                # Иначе это прогноз, и смешивать их в одном ряду нельзя.
                "actual": bool(vintage is not None and end < vintage),
                "production": _number(balance.rows[ROW_PRODUCTION].cells.get(index)),
                "consumption": _number(balance.rows[ROW_CONSUMPTION].cells.get(index)),
                "stock_change": _number(balance.rows[ROW_STOCK_CHANGE].cells.get(index)),
                "oecd_inventories": _number(balance.rows[ROW_OECD_INVENTORIES].cells.get(index)),
                "brent": _number(brent.cells.get(index)) if brent is not None else None,
            }
        )
    return pd.DataFrame(records)


def load_all_vintages(reports_dir: str | None = None) -> pd.DataFrame:
    """Все наблюдения всех выпусков, без выбора лучшего.

    Нужна там, где интересны именно РАСХОЖДЕНИЯ между выпусками: прогноз EIA на
    квартал вперёд и то, чем этот квартал оказался на самом деле.
    """
    directory = reports_dir or settings.reports_dir
    frames = []
    for path in sorted(glob.glob(os.path.join(directory, "*.pdf"))):
        try:
            frame = read_report_factors(path)
        except Exception:  # noqa: BLE001
            continue
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def projection_error_sigma(reports_dir: str | None = None) -> float | None:
    """Насколько прогноз добычи на ближайший квартал расходится с фактом.

    ★Величина нужна для ЧЕСТНОЙ ширины коридора у факторного метода. Его полоса,
    посчитанная по одним лишь остаткам регрессии, отвечает на вопрос «насколько
    точно модель переводит факторы в цену» — и молчит о том, что сами факторы
    взяты из ПРОГНОЗА EIA и тоже могут не сбыться. Второй источник ошибки не
    меньше первого, и не учитывать его значит рисовать уверенность, которой нет.

    Измеряется по нашему же корпусу: у каждого квартала есть прогнозное значение
    из более раннего выпуска и фактическое из более позднего. Возвращается
    стандартное отклонение логарифмической разницы, или ``None``, если пар
    слишком мало.
    """
    everything = load_all_vintages(reports_dir)
    if everything.empty:
        return None

    actual = (
        everything[everything["actual"]]
        .sort_values("vintage")
        .drop_duplicates(subset=["quarter_end"], keep="last")
        .set_index("quarter_end")["production"]
    )
    projected = everything[~everything["actual"]].dropna(subset=["production"])

    errors = []
    for _, row in projected.iterrows():
        # Только ближайший к выпуску квартал: чем дальше прогноз, тем больше
        # ошибка, и смешивать горизонты в одну сигму значило бы получить число,
        # не относящееся ни к одному из них.
        if (row["quarter_end"] - row["vintage"]).days > 120:
            continue
        truth = actual.get(row["quarter_end"])
        if truth is None or not np.isfinite(truth) or truth <= 0 or row["production"] <= 0:
            continue
        errors.append(float(np.log(truth) - np.log(row["production"])))

    if len(errors) < 3:
        return None
    return float(np.std(errors, ddof=1))


def load_factors(reports_dir: str | None = None) -> FactorSeries:
    """Собрать панель факторов по всему корпусу отчётов.

    Для каждого квартала берётся САМЫЙ СВЕЖИЙ выпуск, в котором этот квартал
    уже фактический; если фактических нет ни в одном — самый свежий прогноз.
    Так история получается настолько пересмотренной, насколько корпус позволяет,
    а будущее — настолько свежим.
    """
    directory = reports_dir or settings.reports_dir
    frames = []
    failed: list[str] = []
    paths = sorted(glob.glob(os.path.join(directory, "*.pdf")))
    for path in paths:
        try:
            frame = read_report_factors(path)
        except Exception:  # noqa: BLE001 — один нечитаемый отчёт не отменяет остальные
            failed.append(os.path.basename(path))
            continue
        if not frame.empty:
            frames.append(frame)
        else:
            failed.append(os.path.basename(path))

    if not frames:
        return FactorSeries(frame=pd.DataFrame(), reports_read=0, reports_failed=failed)

    everything = pd.concat(frames, ignore_index=True)
    # Сортировка ставит в конец группы предпочтительное наблюдение: сначала
    # фактические, среди них — самое свежее по выпуску. keep="last" его и берёт.
    everything = everything.sort_values(["quarter_end", "actual", "vintage"])
    chosen = everything.drop_duplicates(subset=["quarter_end"], keep="last")
    chosen = chosen.sort_values("quarter_end").set_index("quarter_end")
    return FactorSeries(
        frame=chosen.drop(columns=["quarter"]),
        reports_read=len(frames),
        reports_failed=failed,
    )
