"""Intake acceptance: what we know about a PDF *before* we believe what it says.

★ЭТОТ МОДУЛЬ НИЧЕГО НЕ ЧИНИТ. Он отвечает на один вопрос: что в этом файле
устроено так, что прочитанному нельзя верить молча. Ответ — ОТЧЁТ, а не
исключение и не тишина: молчание неотличимо от «проверка не запускалась».

Почему это первый шаг, а не удобство. Корпус EIA STEO структурно чист — у всех
восьми файлов на месте `%%EOF` и `startxref`, `pdfinfo` молчит, число страниц
сходится. И при этом извлечение давало 79% строк с ЧУЖИМ заголовком таблицы
(порядок чтения) и несёт шрифты без ToUnicode (декодирование). Отсюда правило,
которым живёт модуль: **«битых файлов нет» ≠ «читается верно»**. Проверять надо
не целость контейнера, а те свойства, из-за которых текст приезжает не тот.

Контракт (он же причина, по которой модуль отдельный):

* **Тотальность.** `inspect_pdf` не поднимает исключений. Любой сбой — включая
  отсутствие внешнего инструмента и его таймаут — становится находкой с явным
  кодом. Недоказанное состояние называется `UNKNOWN`, а не выдаётся за «ок».
* **Детерминизм, а не правильность.** Один и тот же файл + одна и та же версия
  модуля дают один и тот же отчёт. Мы не утверждаем, что нашли все дефекты, —
  мы утверждаем, что перечисленные проверки выполнены и их исход воспроизводим.
* **Тонкий шов.** Ни одна проверка не встроена в конвейер: единственная точка
  входа — функция. Позже она заменится вызовом раст-библиотеки (exlayout), и
  заменить придётся один файл, а не тракт приёма документов.

Лестница болезни (операционная — «сколько усилий на восстановление логики»):

    OK      контейнер цел, дефектов чтения не видно
    NOTICE  читается, но что-то стоит знать (нет ToUnicode, пустые страницы)
    WARN    читается частично или разные пути дают разный результат
    BROKEN  контейнер нарушен: чтение возможно только режимом восстановления
    UNKNOWN проверку выполнить не удалось — статус неизвестен, а не «чисто»
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field

__all__ = [
    "OK", "NOTICE", "WARN", "BROKEN", "UNKNOWN",
    "SEVERITY_ORDER", "Finding", "IntakeReport",
    "inspect_pdf", "inspect_directory",
]

OK = "ok"
NOTICE = "notice"
WARN = "warn"
BROKEN = "broken"
UNKNOWN = "unknown"

# Порядок ступеней задан ЯВНО списком, а не сравнением строк: худшая ступень
# отчёта — максимум по этому порядку, и он обязан быть частью контракта, а не
# следствием алфавита.
SEVERITY_ORDER = [OK, NOTICE, UNKNOWN, WARN, BROKEN]

# Хвост, в котором ищется `%%EOF`. Спецификация разрешает мусор после него,
# практика — тоже; 4 КиБ покрывают штатные подписи и оболочки, но не позволяют
# признать здоровым файл, у которого за концом висит килобайты постороннего.
_TAIL_BYTES = 4096
# Заголовок `%PDF-` по спецификации может стоять не в нулевом байте.
_HEAD_BYTES = 1024
# Внешние инструменты ограничены по времени. Таймаут здесь НЕ вносит
# недетерминизма в вердикт: его исход — фиксированная находка UNKNOWN, а не
# «как повезёт».
_TOOL_TIMEOUT = 30


@dataclass(frozen=True)
class Finding:
    """One observation about the file. ``code`` is stable; ``detail`` is prose."""

    code: str
    severity: str
    detail: str

    def to_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity, "detail": self.detail}


@dataclass
class IntakeReport:
    path: str
    findings: list[Finding] = field(default_factory=list)
    facts: dict = field(default_factory=dict)

    def add(self, code: str, severity: str, detail: str) -> None:
        self.findings.append(Finding(code, severity, detail))

    @property
    def severity(self) -> str:
        """Worst step reached. ``OK`` only when every check ran and found nothing."""
        worst = OK
        for finding in self.findings:
            if SEVERITY_ORDER.index(finding.severity) > SEVERITY_ORDER.index(worst):
                worst = finding.severity
        return worst

    @property
    def readable(self) -> bool:
        """Whether the normal reading path may be trusted without a caveat."""
        return self.severity in (OK, NOTICE)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "severity": self.severity,
            "readable": self.readable,
            "facts": self.facts,
            "findings": [f.to_dict() for f in self.findings],
        }

    def summary(self) -> str:
        head = f"{os.path.basename(self.path)}: {self.severity}"
        if not self.findings:
            return head
        return head + "\n" + "\n".join(
            f"  [{f.severity}] {f.code}: {f.detail}" for f in self.findings
        )


def _run(argv: list[str]) -> tuple[int, str, str] | None:
    """Run an external tool. ``None`` means it is absent or did not finish."""
    if shutil.which(argv[0]) is None:
        return None
    try:
        done = subprocess.run(
            argv, capture_output=True, text=True, timeout=_TOOL_TIMEOUT, check=False
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    return done.returncode, done.stdout, done.stderr


def _check_container(report: IntakeReport, data: bytes) -> None:
    """Byte-level checks. These need no library and cannot be fooled by one."""
    report.facts["size"] = len(data)

    if not data:
        report.add("empty", BROKEN, "файл пуст")
        return

    if b"%PDF-" not in data[:_HEAD_BYTES]:
        report.add("no_header", BROKEN, "нет сигнатуры %PDF- в первом килобайте")
    else:
        version = re.search(rb"%PDF-(\d+\.\d+)", data[:_HEAD_BYTES])
        if version:
            report.facts["version"] = version.group(1).decode("ascii")

    tail = data[-_TAIL_BYTES:]
    if b"%%EOF" not in tail:
        # Самый частый вид порчи в дикой природе — усечение при выкачке
        # (22% в Common Crawl). Он же самый дешёвый в детекции.
        report.add("no_eof", BROKEN, "нет %%EOF в хвосте — вероятно, файл усечён")

    # Число `%%EOF` по всему файлу = число ревизий. Инкрементальные обновления
    # законны, но означают, что «последнее слово» об объекте берётся не из
    # первой таблицы; для нас это признак, а не дефект.
    revisions = data.count(b"%%EOF")
    report.facts["revisions"] = revisions
    if revisions > 1:
        report.add(
            "incremental_updates", NOTICE,
            f"инкрементальных ревизий: {revisions} — актуальные объекты в последней",
        )

    positions = [m.start() for m in re.finditer(rb"startxref", data)]
    if not positions:
        report.add("no_startxref", BROKEN, "нет startxref — таблицу ссылок надо искать сканом")
    else:
        # Проверяется только ПОСЛЕДНИЙ startxref: он и есть точка входа.
        offset = re.search(rb"startxref\s+(\d+)", data[positions[-1]:])
        if offset is None:
            report.add("bad_startxref", BROKEN, "после startxref нет числа")
        else:
            value = int(offset.group(1))
            report.facts["startxref"] = value
            if value <= 0 or value >= len(data):
                report.add(
                    "startxref_out_of_range", BROKEN,
                    f"startxref указывает на {value} при длине файла {len(data)}",
                )
            elif not data[value:value + 32].lstrip().startswith((b"xref", b"%", b"<<")) \
                    and not re.match(rb"\s*\d+\s+\d+\s+obj", data[value:value + 32]):
                # Смещение внутри файла, но ведёт не туда: типичный след
                # переупаковки файла инструментом, не пересчитавшим таблицу.
                report.add(
                    "startxref_misaligned", WARN,
                    f"по смещению {value} нет ни xref, ни объекта",
                )

    if b"/Encrypt" in data:
        report.add("encrypted", WARN, "документ помечен как зашифрованный")


def _check_pdfinfo(report: IntakeReport) -> int | None:
    """Page count according to poppler, plus whatever it complains about."""
    result = _run(["pdfinfo", report.path])
    if result is None:
        report.add("pdfinfo_unavailable", UNKNOWN, "pdfinfo недоступен — проверка не выполнена")
        return None

    code, out, err = result
    if code != 0:
        report.add("pdfinfo_failed", WARN, f"pdfinfo вернул {code}: {err.strip()[:200]}")
        return None
    if err.strip():
        # ★ЖАЛОБА В stderr ПРИ КОДЕ 0 — ЭТО НАХОДКА, А НЕ ШУМ. Poppler чинит
        # многое молча и сообщает об этом только сюда; если смотреть на код
        # возврата, восстановленный файл неотличим от целого.
        report.add("pdfinfo_warnings", WARN, f"pdfinfo пишет в stderr: {err.strip()[:200]}")

    pages = re.search(r"^Pages:\s+(\d+)", out, re.MULTILINE)
    if pages is None:
        report.add("no_page_count", WARN, "pdfinfo не сообщил число страниц")
        return None
    count = int(pages.group(1))
    report.facts["pages_pdfinfo"] = count
    return count


def _check_fonts(report: IntakeReport) -> None:
    """Fonts without a ToUnicode map decode to the wrong characters, silently."""
    result = _run(["pdffonts", report.path])
    if result is None:
        report.add("pdffonts_unavailable", UNKNOWN, "pdffonts недоступен — проверка не выполнена")
        return

    code, out, _err = result
    if code != 0:
        report.add("pdffonts_failed", UNKNOWN, f"pdffonts вернул {code}")
        return

    lines = [line for line in out.splitlines()[2:] if line.strip()]
    if not lines:
        report.facts["fonts_total"] = 0
        return

    # Колонки pdffonts: name type encoding emb sub uni object ID. Столбец `uni`
    # — предпоследний перед номером объекта; берём с конца, потому что имя
    # шрифта содержит пробелы, а хвост фиксирован.
    without_unicode = []
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        uni = parts[-3]
        if uni == "no":
            without_unicode.append(parts[0])
    report.facts["fonts_total"] = len(lines)
    report.facts["fonts_without_tounicode"] = len(without_unicode)
    if without_unicode:
        # Семейство Font-Decoding Split: текст извлекается, выглядит текстом, но
        # символы не те. Ни одна проверка целости этого не видит.
        report.add(
            "fonts_without_tounicode", NOTICE,
            f"{len(without_unicode)} из {len(lines)} шрифтов без ToUnicode "
            f"({', '.join(without_unicode[:3])}…) — риск неверного декодирования",
        )


def _check_text_layer(report: IntakeReport, pages_expected: int | None) -> None:
    """Does our own reading path see the same document poppler does?"""
    try:
        from pdf2xml import parse_pdf

        document = parse_pdf(report.path)
        pages = list(document.pages)
    except Exception as exc:  # noqa: BLE001 — тотальность важнее разбора причин
        report.add("extract_failed", WARN, f"pdf2xml не разобрал файл: {type(exc).__name__}: {exc}")
        return

    report.facts["pages_extractor"] = len(pages)
    empty = [page.number for page in pages if not page.text().strip()]
    report.facts["pages_without_text"] = len(empty)

    if pages and len(empty) == len(pages):
        report.add("no_text_layer", WARN, "ни на одной странице нет текста — вероятно, скан")
    elif empty:
        report.add(
            "pages_without_text", NOTICE,
            f"страниц без текста: {len(empty)} из {len(pages)} "
            f"(№ {', '.join(str(n) for n in empty[:5])}{'…' if len(empty) > 5 else ''})",
        )

    if pages_expected is not None and pages_expected != len(pages):
        # ★ДВА ПУТИ РАСХОДЯТСЯ — ЗНАЧИТ, ХОТЯ БЫ ОДИН НЕВЕРЕН. Какой именно,
        # приёмка не решает: её дело — не пропустить расхождение дальше молча.
        report.add(
            "page_count_mismatch", WARN,
            f"pdfinfo насчитал {pages_expected} страниц, извлекатель — {len(pages)}",
        )


def inspect_pdf(path: str) -> IntakeReport:
    """Inspect one file. Never raises; the report always answers."""
    report = IntakeReport(path=path)

    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        report.add("unreadable", BROKEN, f"файл не читается: {exc}")
        return report

    _check_container(report, data)
    # Внешние инструменты запускаются, даже если контейнер уже признан битым:
    # частичное чтение — законный исход, и знать, сколько именно удалось
    # прочесть, полезнее, чем остановиться на первой находке.
    pages_expected = _check_pdfinfo(report)
    _check_fonts(report)
    _check_text_layer(report, pages_expected)
    return report


def inspect_directory(directory: str) -> list[IntakeReport]:
    """Inspect every PDF in a directory, in sorted order (so runs compare)."""
    if not os.path.isdir(directory):
        return []
    reports = []
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith(".pdf"):
            reports.append(inspect_pdf(os.path.join(directory, name)))
    return reports


if __name__ == "__main__":  # pragma: no cover — ручной прогон по каталогу
    import sys

    targets = sys.argv[1:] or ["."]
    for target in targets:
        items = inspect_directory(target) if os.path.isdir(target) else [inspect_pdf(target)]
        for item in items:
            print(item.summary())
