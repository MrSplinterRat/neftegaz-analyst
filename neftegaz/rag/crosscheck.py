"""Differential cross-check: two readers over the same page.

★ЗАЧЕМ ВТОРОЙ ЧИТАТЕЛЬ. У одного извлекателя нет способа узнать, что он ошибся:
он отдаёт текст, текст выглядит текстом, и никакая проверка целости файла этого
не видит. Отказ, о котором мы говорим, — тихий. Единственный дешёвый свидетель
против такого отказа — ВТОРОЙ путь чтения, устроенный иначе: там, где два
независимых читателя согласны, ошибиться могут только оба сразу; там, где они
расходятся, неверен как минимум один — и это место надо назвать, а не замять.

Свидетель обязан быть независим от проверяемого. Здесь это выполнено ЧАСТИЧНО, и
граница проходит ровно по слою декодирования:

    poppler   pdftotext -layout        свой декодер, сборка по операторам
    pdf2xml   pdftotext -bbox-layout   ТОТ ЖЕ декодер, сборка по геометрии
    pypdf     чистый питон             ДРУГОЙ декодер, сборка по операторам

★Первые двое — один и тот же бинарь poppler. Они расходятся в том, что делают
ПОСЛЕ декодирования глифов, и потому свидетельствуют только о сборке: порядок
чтения, склейка токенов, потерянная колонка. К искажению самого декодирования
они слепы ОБА СРАЗУ: если `ToUnicode` шрифта врёт, оба пути врут одинаково и
дружно скажут AGREE. Это ровно та двойная слепота, ради которой добавлен третий
читатель: pypdf разбирает поток содержимого и раскодирует глифы своим кодом, не
разделяя с poppler ни строчки. Расхождение с ним свидетельствует НИЖЕ сборки.

Совпадение по-прежнему не доказывает правоты — оно лишь исчерпывает то, что мы
умеем спросить. Сверка ловит расхождение путей, а не истину; чем независимее
пути, тем шире класс отказов, который она делает громким.

★ЧТО СРАВНИВАЕТСЯ — ЧИСЛА, А НЕ ТЕКСТ. Для аналитика содержание отчёта EIA —
это цифры; слово, приехавшее с опечаткой, стоит несравнимо дешевле, чем цифра,
приехавшая из соседней колонки. Поэтому основная мера — МУЛЬТИМНОЖЕСТВО чисел
страницы, и оно же даёт бесплатное различение двух разных бед:

    числа совпали, порядок совпал      → пути согласны             (AGREE)
    числа совпали, порядок разный      → расхождение ПОРЯДКА ЧТЕНИЯ (ORDER)
    те же цифры, иное членение         → расхождение ГРАНИЦ ТОКЕНОВ (TOKENIZE)
    цифры разошлись                    → ПОТЕРЯ или ПОДМЕНА данных  (DIVERGE)

Различение существенно. ORDER — наш известный случай (заголовок таблицы после
её первой строки на 168 страницах из 208): данные все на месте, неверна их
привязка, и лечится это сборкой по геометрии. DIVERGE — другая болезнь: цифра
либо потеряна, либо декодирована не тем шрифтом, и сборка по геометрии её не
вернёт. Складывать их в одну метрику «похожести» значило бы стереть разницу
между «переставлено» и «утрачено».

★ТРЕТИЙ КЛАСС ПОЯВИЛСЯ ОТ ПЕРВОГО ЖЕ ПРОГОНА ПО КОРПУСУ и стоит того, чтобы
быть названным. Подписи осей на графиках приезжают к poppler слипшимися:
`20242024` там, где на бумаге стоят два соседних `2024`, `00` вместо двух
нулей. Это не потеря — цифры те же и в том же порядке, различается ЧЛЕНЕНИЕ.
Признак проверяется точно: склейка неразошедшихся остатков совпадает посимвольно.
Отдельный класс, а не «согласны»: для аналитика `1234` вместо `12` и `34` —
всё ещё неверное число, просто болезнь другая и лечится не тем же.

Модуль тотален так же, как приёмка: не поднимает исключений, отсутствие или
отказ любого из читателей становится находкой со статусом, а не тишиной.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field

__all__ = [
    "AGREE",
    "ORDER",
    "TOKENIZE",
    "DIVERGE",
    "UNAVAILABLE",
    "VERDICT_ORDER",
    "PageDiff",
    "CrossCheckReport",
    "numbers_in",
    "compare_pages",
    "compare_pages_multi",
    "crosscheck_pdf",
    "crosscheck_directory",
]

AGREE = "agree"
ORDER = "order"
TOKENIZE = "tokenize"
DIVERGE = "diverge"
UNAVAILABLE = "unavailable"

# Порядок тяжести — СПИСКОМ, а не алфавитом и не значением enum. Сводный
# вердикт страницы есть худший из попарных, и «худший» обязан быть определён
# явно: любой неявный порядок однажды переставят, не заметив.
VERDICT_ORDER = [AGREE, ORDER, TOKENIZE, DIVERGE]


def _worst(verdicts: list[str]) -> str:
    return max(verdicts, key=VERDICT_ORDER.index) if verdicts else AGREE


_TOOL_TIMEOUT = 120

# Число в отчётах: необязательный знак, разряды через запятую, дробная часть.
# Требование границы слева и справа отсекает куски идентификаторов и дат вида
# 2026-01 (они разберутся на два числа — и это верно: оба пути увидят их
# одинаково, а нам важно СРАВНЕНИЕ, а не разбор семантики).
_NUMBER_RE = re.compile(
    r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w])"
)


def numbers_in(text: str) -> list[str]:
    """Numeric tokens of a page, normalised, in reading order.

    Нормализация минимальна и обратима на глаз: снимается разделитель разрядов
    и ведущий плюс. Скобочная запись отрицательных чисел НЕ разворачивается —
    оба читателя видят скобки одинаково, а разворачивать её значило бы вносить
    в сверку собственную интерпретацию, которой у читателей нет.
    """
    out = []
    for token in _NUMBER_RE.findall(text):
        token = token.replace(",", "")
        if token.startswith("+"):
            token = token[1:]
        out.append(token)
    return out


@dataclass
class PageDiff:
    page: int
    verdict: str
    numbers_a: int = 0
    numbers_b: int = 0
    only_a: list[str] = field(default_factory=list)
    only_b: list[str] = field(default_factory=list)
    # ── третий читатель и всё, что он делает возможным ──────────────────────
    # Попарные вердикты по именам читателей: {"poppler|pdf2xml": "agree", …}.
    # Хранятся именно все пары, а не сводка: сводка выводима из пар, обратно —
    # нет, и разбирать потом «кто с кем не сошёлся» больше будет неоткуда.
    pairs: dict[str, str] = field(default_factory=dict)
    # ★ГЛАВНЫЙ ВЫИГРЫШ ТРЁХ ПУТЕЙ: имя читателя, который разошёлся с ОБОИМИ
    # остальными, когда те двое согласны. При двух путях известно лишь «кто-то
    # неправ»; при трёх — обычно ВИДНО КТО, а значит расхождение перестаёт быть
    # поводом не доверять странице целиком. `None` — большинства нет.
    outlier: str | None = None

    @property
    def lost(self) -> int:
        """How many numeric tokens one reader sees and the other does not."""
        return len(self.only_a) + len(self.only_b)

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "verdict": self.verdict,
            "numbers_a": self.numbers_a,
            "numbers_b": self.numbers_b,
            "only_a": self.only_a[:20],
            "only_b": self.only_b[:20],
            "lost": self.lost,
            "pairs": self.pairs,
            "outlier": self.outlier,
        }


@dataclass
class CrossCheckReport:
    path: str
    reader_a: str = "poppler"
    reader_b: str = "pdf2xml"
    # Кто реально участвовал в этом прогоне. Отдельно от `reader_a`/`reader_b`,
    # потому что состав читателей теперь зависит от машины: пакета может не быть.
    readers: list[str] = field(default_factory=list)
    pages: list[PageDiff] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def counts(self) -> dict:
        tally = Counter(p.verdict for p in self.pages)
        outliers = Counter(p.outlier for p in self.pages if p.outlier)
        return {
            "pages": len(self.pages),
            AGREE: tally[AGREE],
            ORDER: tally[ORDER],
            TOKENIZE: tally[TOKENIZE],
            DIVERGE: tally[DIVERGE],
            # Потерянным считается только то, что расходится ПО ЦИФРАМ: остаток
            # склейки не потерян, он лежит в соседнем токене.
            "numbers_lost": sum(p.lost for p in self.pages if p.verdict == DIVERGE),
            # Сколько страниц удалось СВЕСТИ К ОДНОМУ ВИНОВНИКУ — и к какому.
            # Это и есть прибавка третьего пути, выраженная числом.
            "localised": sum(outliers.values()),
            "outliers": dict(outliers),
        }

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "readers": self.readers or [self.reader_a, self.reader_b],
            "counts": self.counts(),
            "notes": self.notes,
            "pages": [p.to_dict() for p in self.pages if p.verdict != AGREE],
        }

    def summary(self) -> str:
        c = self.counts()
        head = (
            f"{os.path.basename(self.path)}: страниц {c['pages']}, "
            f"согласны {c[AGREE]}, порядок {c[ORDER]}, склейка {c[TOKENIZE]}, "
            f"цифры расходятся {c[DIVERGE]} (потеряно токенов {c['numbers_lost']})"
        )
        lines = [head]
        if c["localised"]:
            who = ", ".join(f"{n}: {k}" for n, k in sorted(c["outliers"].items()))
            lines.append(f"  сведено к одному пути на {c['localised']} стр. — {who}")
        lines += [f"  ! {note}" for note in self.notes]
        for page in self.pages:
            if page.verdict == DIVERGE:
                lines.append(
                    f"  стр. {page.page}: только у {self.reader_a}: "
                    f"{', '.join(page.only_a[:6])}{'…' if len(page.only_a) > 6 else ''} | "
                    f"только у {self.reader_b}: "
                    f"{', '.join(page.only_b[:6])}{'…' if len(page.only_b) > 6 else ''}"
                )
        return "\n".join(lines)


def _digit_counts(tokens: list[str]) -> Counter:
    """How many of each digit the page holds, ignoring where the tokens break."""
    return Counter(ch for token in tokens for ch in token if ch.isdigit())


def _residual(seq: list[str], shared: Counter) -> list[str]:
    """What is left of ``seq`` after removing the tokens both readers agree on.

    Порядок появления сохраняется намеренно: по нему проверяется склейка, а
    отсортированный остаток эту проверку сделал бы невозможной.
    """
    budget = Counter(shared)
    rest = []
    for token in seq:
        if budget[token] > 0:
            budget[token] -= 1
        else:
            rest.append(token)
    return rest


def compare_pages(text_a: str, text_b: str, page: int) -> PageDiff:
    """The whole verdict logic, on plain strings — so it is testable without a PDF."""
    a, b = numbers_in(text_a), numbers_in(text_b)
    count_a, count_b = Counter(a), Counter(b)

    if count_a == count_b:
        verdict = AGREE if a == b else ORDER
        return PageDiff(page=page, verdict=verdict, numbers_a=len(a), numbers_b=len(b))

    shared = count_a & count_b
    only_a = _residual(a, shared)
    only_b = _residual(b, shared)

    # Склейка проверяется по МУЛЬТИМНОЖЕСТВУ ЦИФР страницы, а не по совпадению
    # склеенных остатков. Первая версия сравнивала остатки строкой — и на
    # корпусе тут же нашлись страницы, где склейка идёт вперемешку с одной
    # настоящей разницей: строгое равенство рушилось, и весь лист уходил в
    # DIVERGE вместе с полусотней безобидных подписей осей. Счётчик цифр к
    # членению и к порядку нечувствителен по построению, поэтому отвечает
    # ровно на свой вопрос: «те же ли это цифры».
    if _digit_counts(a) == _digit_counts(b):
        verdict = TOKENIZE
    else:
        verdict = DIVERGE

    return PageDiff(
        page=page,
        verdict=verdict,
        numbers_a=len(a),
        numbers_b=len(b),
        only_a=only_a,
        only_b=only_b,
    )


def reduce_to_comparable(verdict: str, positional: bool) -> str:
    """Оставить от вердикта пары только то, о чём эта пара вправе судить.

    ★ПАРА ОТВЕЧАЕТ НЕ НА ВСЕ ВОПРОСЫ. `ORDER` и `TOKENIZE` — суждения о том, в
    каком порядке и какими кусками текст лёг на страницу. Такое суждение
    осмысленно, только если ОБА читателя строят последовательность от ПОЛОЖЕНИЯ
    на странице: poppler в режиме `-layout` и pdf2xml по координатам — строят.
    pypdf выдаёт текст в порядке операторов потока содержимого и о колонках не
    знает ничего; его «другой порядок» — не находка, а устройство.

    Замерено: против pypdf `ORDER` срабатывает на 322 страницах из 469, то есть
    почти всегда. Признак, который зажигается почти всегда, не несёт сведений —
    он лишь заглушает те, что несут. Поэтому для пар с непозиционным читателем
    остаётся ровно один вопрос, на который он отвечает лучше всех: ТЕ ЖЕ ЛИ
    ЦИФРЫ. Он и есть вопрос к декодеру, ради которого третий путь добавлен.
    """
    if positional or verdict == DIVERGE:
        return verdict
    return AGREE


def compare_pages_multi(
    texts: dict[str, str],
    page: int,
    primary: tuple[str, str],
    positional: dict[str, bool] | None = None,
) -> PageDiff:
    """Verdict of one page over ANY number of readers.

    Считаются ВСЕ пары. Сводный вердикт страницы — худший из попарных: страница
    не может считаться прочитанной лучше, чем худшее наблюдаемое расхождение.

    ★Выброс (`outlier`) ищется по определению «разошёлся с обоими, а те двое
    согласны». Это не голосование за истину: большинство может ошибаться хором,
    и при двух читателях из трёх на общем декодере ошибётся именно хором. Это
    указание, ГДЕ смотреть, — и уже оно превращает «странице нельзя верить» в
    «вот этот путь на этой странице читает не так».

    `primary` — пара, чьи подробности едут в `only_a`/`only_b`: сводка остаётся
    читаемой и сравнимой с прежними прогонами, когда путей было два.
    """
    names = sorted(texts)
    pos = positional or {}
    pairs: dict[str, str] = {}
    detail: dict[str, PageDiff] = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            diff = compare_pages(texts[left], texts[right], page)
            both_positional = pos.get(left, True) and pos.get(right, True)
            pairs[f"{left}|{right}"] = reduce_to_comparable(diff.verdict, both_positional)
            detail[f"{left}|{right}"] = diff

    def key(x: str, y: str) -> str:
        return f"{x}|{y}" if x < y else f"{y}|{x}"

    # ★ВЫБРОС ОПРЕДЕЛЁН ЧЕРЕЗ СРАВНЕНИЕ, А НЕ ЧЕРЕЗ ПОЛНОЕ СОГЛАСИЕ. Первая
    # редакция требовала, чтобы двое остальных сошлись ДОСЛОВНО (`agree`), и на
    # корпусе почти не срабатывала: наша главная пара часто расходится по
    # порядку или членению, и страница с безобидным `order` между двумя
    # позиционными путями всё равно уходила в спорные из-за третьего. Между тем
    # смысл выброса — «этот разошёлся с обоими СИЛЬНЕЕ, чем они между собой»,
    # и именно так теперь и проверяется: пара остальных строго лучше каждой
    # пары с подозреваемым.
    def rank(verdict: str) -> int:
        return VERDICT_ORDER.index(verdict)

    outlier = None
    if len(names) >= 3:
        for suspect in names:
            others = [n for n in names if n != suspect]
            between_others = max(
                (rank(pairs[key(a, b)]) for i, a in enumerate(others) for b in others[i + 1 :]),
                default=0,
            )
            worse_with_suspect = min(rank(pairs[key(suspect, other)]) for other in others)
            if worse_with_suspect > between_others:
                outlier = suspect
                break

    base = detail.get(key(*primary))
    return PageDiff(
        page=page,
        verdict=_worst(list(pairs.values())),
        numbers_a=base.numbers_a if base else 0,
        numbers_b=base.numbers_b if base else 0,
        only_a=base.only_a if base else [],
        only_b=base.only_b if base else [],
        pairs=pairs,
        outlier=outlier,
    )


def _read_poppler(path: str) -> list[str] | None:
    """Pages as poppler sees them. ``None`` when poppler cannot answer."""
    if shutil.which("pdftotext") is None:
        return None
    try:
        done = subprocess.run(
            ["pdftotext", "-layout", path, "-"],
            capture_output=True,
            timeout=_TOOL_TIMEOUT,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if done.returncode != 0:
        return None
    # Разделитель страниц у pdftotext — form feed. Хвостовой \f даёт пустой
    # элемент, его снимаем, иначе последняя «страница» будет фантомной.
    text = done.stdout.decode("utf-8", errors="replace")
    pages = text.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    return pages


def _read_extractor(path: str) -> list[str] | None:
    try:
        from pdf2xml import parse_pdf

        # ★dedupe=False — СВЕРКА СМОТРИТ НА ЧТЕНИЕ, А НЕ НА ПОЧИНКУ.
        # Снятие двойной отрисовки — наша правка поверх прочитанного; poppler
        # её не делает и делать не может (в режиме -layout координат нет).
        # Сверяя починенное с непочиненным, мы получили бы расхождение на всех
        # страницах, где починка сработала, — а их 292 из 469. Замерено: так
        # «цифры расходятся» выросло с 49 до 122 страниц, и ни одна из этих 73
        # не была ошибкой чтения. Починка проверяется отдельно (102 723 ячейки
        # таблиц, расхождений ноль) и другим механизмом.
        return [page.text(dedupe=False) for page in parse_pdf(path).pages]
    except Exception:  # noqa: BLE001 — тотальность важнее разбора причин
        return None


def _read_pypdf(path: str) -> list[str] | None:
    """Pages as pypdf sees them — ДРУГИМ декодером, не поплеровским.

    Ради этого он и добавлен: pypdf разбирает поток содержимого и раскладывает
    глифы в символы своим кодом. Там, где два поплеровских пути слепы одинаково
    (кривой `ToUnicode`, подменённый встроенный шрифт), этот путь видит своё.
    Расплата — другая сборка строк: он не знает про колонки, поэтому по ПОРЯДКУ
    расходится с геометрией часто и законно. Именно поэтому классы вердиктов
    разделены: ORDER от него ожидаем, DIVERGE — нет.
    """
    try:
        from pypdf import PdfReader

        return [(page.extract_text() or "") for page in PdfReader(path).pages]
    except Exception:  # noqa: BLE001 — тотальность важнее разбора причин
        return None


# Читатели перечислены СПИСКОМ, а не собраны по месту: добавить четвёртый путь
# должно стоить одну строку здесь, иначе расширение упрётся в правку логики.
# Третье поле — ПОЗИЦИОННОСТЬ: строит ли читатель последовательность текста от
# положения на странице. От неё зависит, о чём пара с этим читателем вправе
# судить (см. `reduce_to_comparable`), и она обязана стоять рядом с читателем, а
# не выводиться где-то по имени.
READERS: list[tuple[str, object, bool]] = [
    ("poppler", _read_poppler, True),
    ("pdf2xml", _read_extractor, True),
    ("pypdf", _read_pypdf, False),
]


def crosscheck_pdf(path: str) -> CrossCheckReport:
    """Compare every available reader page by page. Never raises.

    Читателя, который не ответил, сверка НЕ ждёт и НЕ подменяет: он выбывает,
    его отсутствие становится примечанием, а оставшиеся сверяются между собой.
    Требовать всех значило бы терять всю проверку из-за неустановленного пакета —
    то есть наказывать за неполноту молчанием вместо ослабленного, но честного
    ответа. Меньше двух ответивших — сверять нечего, и это тоже сказано вслух.
    """
    report = CrossCheckReport(path=path)

    pages: dict[str, list[str]] = {}
    positional: dict[str, bool] = {}
    for name, read, is_positional in READERS:
        got = read(path)
        if got is None:
            report.notes.append(f"читатель не ответил: {name} — его путь в сверке не участвует")
        else:
            pages[name] = got
            positional[name] = is_positional

    if len(pages) < 2:
        report.notes.append("ответил меньше двух читателей — сверка не выполнена")
        return report

    report.readers = sorted(pages)
    counts = {name: len(seq) for name, seq in pages.items()}
    common = min(counts.values())
    if len(set(counts.values())) > 1:
        # ★РАСХОЖДЕНИЕ ПО ЧИСЛУ СТРАНИЦ — САМО ПО СЕБЕ НАХОДКА, и она не
        # отменяет сверку: общий префикс сравнить всё равно можно и нужно.
        listed = ", ".join(f"{n} {c}" for n, c in sorted(counts.items()))
        report.notes.append(f"разное число страниц: {listed} — сверены первые {common}")

    primary = (report.reader_a, report.reader_b)
    for index in range(common):
        report.pages.append(
            compare_pages_multi(
                {name: seq[index] for name, seq in pages.items()},
                page=index + 1,
                primary=primary,
                positional=positional,
            )
        )
    return report


def crosscheck_directory(directory: str) -> list[CrossCheckReport]:
    if not os.path.isdir(directory):
        return []
    return [
        crosscheck_pdf(os.path.join(directory, name))
        for name in sorted(os.listdir(directory))
        if name.lower().endswith(".pdf")
    ]


if __name__ == "__main__":  # pragma: no cover — ручной прогон
    import sys

    targets = sys.argv[1:] or ["."]
    for target in targets:
        items = crosscheck_directory(target) if os.path.isdir(target) else [crosscheck_pdf(target)]
        for item in items:
            print(item.summary())
