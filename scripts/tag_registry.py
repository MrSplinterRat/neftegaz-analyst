"""Разметка реестра решений тегами и сборка указателя по тегам.

★Зачем инструмент, а не разметка руками.

Записей 126. Проставить теги глазами — это час работы и второй источник истины:
указатель, набранный отдельно от корпуса, разойдётся с ним ровно так же, как
разошёлся сам реестр с кодом (см. раздел «Как поддерживать этот документ»).
Здесь теги вычисляются ИЗ ТЕЛА ЗАПИСИ, а указатель собирается из проставленных
тегов, поэтому разойтись им не с чем.

★Погрешность принята сознательно (решение владельца проекта 31.08.2026):
машинная разметка по ключевым словам ошибается — приписывает лишнее там, где
слово попало в текст мимоходом, и пропускает там, где предмет назван иначе.
Цена ручной точности выше пользы от неё, поэтому в самом реестре сказано, что
первичная разметка машинная. Это та же честность, что и с полем
«Статус обоснования»: помета о происхождении знания важнее его гладкости.

Правило словаря — ЗАКРЫТЫЙ список: тег заводится не тогда, когда хочется
точности, а когда набралось хотя бы три записи. Тег на одну запись ничего не
группирует, он лишь длинное имя этой записи.

Запуск:
    python scripts/tag_registry.py           # отчёт, файл не трогается
    python scripts/tag_registry.py --apply   # проставить теги и собрать указатель
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent.parent / "docs" / "РЕЕСТР-РЕШЕНИЙ.md"

# Порог различающей силы: тег, накрывающий больше этой доли корпуса, не сужает
# поиск, а лишь изображает его. Порог назван здесь числом, чтобы отчёт мог на
# него сослаться, а не чтобы спрятать.
TOO_BROAD = 0.40
MIN_RECORDS = 3

# ── Фасет «предмет»: единственный, который нельзя вывести из полей записи ────
# Ключи — теги, значения — то, по чему предмет опознаётся в тексте. Регистр не
# учитывается; ищется вхождение, а не слово целиком, потому что русские
# окончания иначе пришлось бы перечислять.
# ⚠Ключи держать УЗКИМИ. Первый прогон 31.08.2026 искал по всему телу записи и
# по широким словам («источник», «ход», «раздел», «данные»): тег «цитирование»
# накрыл 126 записей из 126, восемь тегов вышли шире 40%, медиана дошла до 8
# тегов на запись. Такой указатель не сужает поиск, а изображает его. Лечится
# двумя вещами разом: узкими ключами и узкой областью поиска (см. assign).
SUBJECT = {
    "разбор-pdf": ["pdf", "разборщик", "геометри", "колонк", "страниц", "глиф"],
    "фрагменты": ["чанк", "фрагмент", "нарезк", "перекрыт"],
    "поиск": ["bm25", "эмбеддинг", "вектор", "ранжир", "qdrant", "поиск по"],
    "промпт": ["промпт", "роль", "реплик"],
    "маршрутизация": ["маршрут", "узел", "узла", "граф", "уточня"],
    "диалог-память": ["истори разговор", "разговор", "чекпойнт", "контекст"],
    "прогноз": [
        "прогноз", "arima", "сглажив", "коридор", "доверительн", "эластичн",
        "остатк", "наблюдени",
    ],
    "расчётный-модуль": ["расчётн", "расчетн", "формул", "сценари"],
    "веб-поиск": ["веб-", "веб ", "duckduckgo"],
    "цитирование": ["цитат", "сверк", "провенанс"],
    "честность-отказа": [
        "отказ", "три состояни", "пуста", "вслух", "молч", "оговорк", "предупрежд",
    ],
    "настройки": ["настройк", ".env", "переменн", "умолчани"],
    "интерфейс": ["интерфейс", "streamlit", "панел", "кнопк", "история запрос"],
    "контейнер": ["контейнер", "образ", "docker", "compose", "healthcheck"],
    "защита": ["защит", "привилег", "tmpfs", "изоляц", "безопасн", "секрет"],
    "приёмка": ["тест", "приёмк", "линтер", "замер", "форматирован"],
    # Решения про рамку самой работы: что считается заданием, что входит в
    # поставку, чем пишется решение. Их мало, но без тега они невидимы.
    "рамка-работы": ["задани", "поставк", "ouroboros", "контуре продукта"],
    "корпус-данные": ["корпус", "eia", "steo", "csv"],
    "языковая-модель": ["языков", "llm", "openai", "gigachat"],
    "локальность": ["локальн", "интернет", "исходящ", "телеметри"],
    "публикация": ["публикац", "репозитор", "лицензи"],
    "документы": ["readme", "отчёт —", "реестр"],
}

# Сколько тегов предмета оставлять записи. Больше четырёх — и запись всплывает
# почти в каждом теге, то есть указатель перестаёт различать.
MAX_SUBJECTS = 4

MARK_START = "<!-- ТЕГИ: начало сгенерированного указателя -->"
MARK_END = "<!-- ТЕГИ: конец сгенерированного указателя -->"

# ── Ссылки между записями ───────────────────────────────────────────────────
# Якорь ставится ЯВНЫЙ, а не берётся из автоматического якоря заголовка. Причина:
# автоматический строится из полного текста заголовка, то есть меняется от любой
# правки формулировки — и все ссылки на запись молча превращаются в никуда.
# Явный якорь привязан к номеру, а номер за решением закреплён навсегда.
ANCHOR = '<a id="r-{n}"></a>'
ANCHOR_RE = re.compile(r'^<a id="r-\d+"></a>\n', re.M)
# Уже проставленная ссылка — чтобы прогон был идемпотентным, они сперва
# разбираются обратно в голый номер, а затем ставятся заново.
LINKED_RE = re.compile(r"\[(Р-\d+)\]\(#r-\d+\)")
NUMBER_RE = re.compile(r"(?<!\[)(Р-(\d+))")


class Record:
    def __init__(self, number: str, title: str, body: str, start: int) -> None:
        self.number = number
        self.title = title
        self.body = body
        self.start = start
        self.subjects: list[str] = []

    @property
    def nature(self) -> str | None:
        """Фасет «природа решения» — читается из поля, а не угадывается."""
        m = re.search(r"\*\*Возможные альтернативы\.\*\*\s*(.{0,80})", self.body, re.S)
        if not m:
            return None
        text = m.group(1).lower()
        for key, tag in (
            ("предписано", "предписано-ТЗ"),
            ("навязано", "навязано-средой"),
            ("владельца", "решение-владельца"),
            ("развилка", "развилка-была"),
            ("не рассматривались", "без-альтернатив"),
            ("отменяет", "пересмотр"),
        ):
            if key in text:
                return tag
        return None

    @property
    def state(self) -> str:
        """Фасет «состояние» — тоже из поля."""
        m = re.search(r"\*\*Статус:\*\*\s*(.{0,40})", self.body)
        if not m:
            return "?"
        text = m.group(1).lower()
        if text.startswith("отменено"):
            return "отменено"
        if "дополнено" in text:
            return "дополнено"
        if "спроектиров" in text:
            return "спроектировано"
        return "действует"

    @property
    def reconstructed(self) -> bool:
        return "**Статус обоснования:** РЕКОНСТРУИРОВАНО" in self.body


def parse(text: str) -> list[Record]:
    lines = text.split("\n")
    records: list[Record] = []
    cur: list[str] = []
    number = title = ""
    start = 0
    for i, line in enumerate(lines):
        m = re.match(r"^### (Р-\d+)\. (.+)$", line)
        if m:
            if number:
                records.append(Record(number, title, "\n".join(cur), start))
            number, title, start, cur = m.group(1), m.group(2), i, []
        elif number:
            if line.startswith(("## ", "# ")):
                records.append(Record(number, title, "\n".join(cur), start))
                number = ""
                cur = []
            else:
                cur.append(line)
    if number:
        records.append(Record(number, title, "\n".join(cur), start))
    return records


def _subject_area(r: Record) -> str:
    """Где искать предмет записи.

    ★Не по всему телу. «Обоснование» и «Провенанс» — это рассуждение и адрес, там
    слова встречаются мимоходом: запись про лицензию упоминает и поиск, и отчёт,
    и приёмку, не будучи ни тем, ни другим, ни третьим. Предмет объявлен в
    заголовке и в двух полях — «Задача» (что решали) и «Выбранная альтернатива»
    (что сделали).
    """
    parts = [r.title]
    for field in ("Задача", "Выбранная альтернатива"):
        m = re.search(rf"\*\*{field}\.\*\*(.*?)(?=\n\*\*|\Z)", r.body, re.S)
        if m:
            parts.append(m.group(1))
    return "\n".join(parts).lower()


def assign(records: list[Record]) -> None:
    for r in records:
        haystack = _subject_area(r)
        # Сила тега — число разных ключей, попавших в текст. При отборе лучших
        # это отличает предмет записи от слова, сказанного вскользь.
        hits = {
            tag: sum(k in haystack for k in keys)
            for tag, keys in SUBJECT.items()
            if any(k in haystack for k in keys)
        }
        best = sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_SUBJECTS]
        r.subjects = sorted(tag for tag, _ in best)


def report(records: list[Record]) -> int:
    total = len(records)
    counts = {tag: sum(tag in r.subjects for r in records) for tag in SUBJECT}
    print(f"записей: {total}")
    print(f"\n{'тег':22} {'записей':>8}  {'доля':>6}  примечание")
    bad = 0
    for tag, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        share = n / total
        note = ""
        if n < MIN_RECORDS:
            note = f"⚠ меньше {MIN_RECORDS} — упразднить или слить"
            bad += 1
        elif share > TOO_BROAD:
            note = f"⚠ шире {TOO_BROAD:.0%} — не сужает поиск"
            bad += 1
        print(f"{tag:22} {n:>8}  {share:>5.0%}  {note}")

    naked = [r.number for r in records if not r.subjects]
    print(f"\nбез единого тега предмета: {len(naked)}" + (f" — {', '.join(naked)}" if naked else ""))
    if naked:
        bad += 1
    per = [len(r.subjects) for r in records]
    per.sort()
    print(f"тегов на запись: медиана {per[len(per) // 2]}, максимум {per[-1]}")
    return bad


def build_index(records: list[Record]) -> str:
    out = [MARK_START, "", "## Указатель по тегам", ""]
    out += [
        "★Разметка предмета **машинная** — по вхождению ключевых слов в текст",
        "записи (`scripts/tag_registry.py`). Она ошибается в обе стороны: приписывает",
        "лишнее, когда слово попало в текст мимоходом, и пропускает, когда предмет",
        "назван иначе. Точность здесь сознательно разменяна на то, что указатель не",
        "может разойтись с корпусом: он собирается из тех же записей, а не набирается",
        "отдельно. Два других фасета ошибаться не могут вовсе — они читаются из полей.",
        "",
        "Запись стои́т под несколькими тегами: это покрытие, а не разбиение. Ссылки, не",
        "копии — текст живёт в одном месте.",
        "",
        "### Предмет — что затронуто",
        "",
    ]
    total = len(records)
    counts = {tag: [r for r in records if tag in r.subjects] for tag in SUBJECT}
    for tag, rs in sorted(counts.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(rs) < MIN_RECORDS:
            continue
        share = len(rs) / total
        wide = "  ⚠тег широк, сужайте вторым тегом" if share > TOO_BROAD else ""
        out.append(f"**{tag}** ({len(rs)}){wide} — " + " · ".join(r.number for r in rs))
        out.append("")

    out += ["### Природа решения — ваш выбор или так вышло", ""]
    natures: dict[str, list[str]] = {}
    for r in records:
        if r.nature:
            natures.setdefault(r.nature, []).append(r.number)
    for tag, nums in sorted(natures.items(), key=lambda kv: -len(kv[1])):
        if len(nums) / total > TOO_BROAD:
            # ★Список не приводится намеренно. Перечень из ста номеров не помогает
            # найти запись — он лишь имитирует указатель. Здесь важно ЧИСЛО: оно
            # говорит, что большинство решений принималось с рассмотренной
            # альтернативой, а это утверждение о работе, а не вход для поиска.
            out.append(
                f"**{tag}** ({len(nums)}, {len(nums) / total:.0%} корпуса) — список не"
                " приводится: тег шире порога различимости и как вход бесполезен."
                " Число здесь и есть содержание."
            )
        else:
            out.append(f"**{tag}** ({len(nums)}) — " + " · ".join(nums))
        out.append("")
    out += [
        "★Тег `без-альтернатив` — это очередь на пересмотр, а не упрёк: места, где",
        "выигрыш может оказаться дешёвым просто потому, что туда никто не смотрел.",
        "",
        "### Состояние",
        "",
    ]
    states: dict[str, list[str]] = {}
    for r in records:
        states.setdefault(r.state, []).append(r.number)
    for tag, nums in sorted(states.items(), key=lambda kv: -len(kv[1])):
        shown = " · ".join(nums) if len(nums) <= 40 else f"{len(nums)} записей"
        out.append(f"**{tag}** ({len(nums)}) — {shown}")
        out.append("")
    rec = [r.number for r in records if r.reconstructed]
    out.append(f"**обоснование восстановлено** ({len(rec)}) — " + " · ".join(rec))
    out += ["", MARK_END]
    return "\n".join(out)


def apply(text: str, records: list[Record]) -> str:
    """Вписать строку тегов в каждую запись и вставить указатель."""
    # ★Вставка идёт ПО УСТРОЙСТВУ шапки записи, а не по угадыванию строк. Шапка —
    # это блок непустых строк, начинающийся с «**Область:**»; поле «Статус» может
    # занимать несколько строк с пояснением, и перечислять их начала (как было в
    # первой версии) значит держать список, который разойдётся с текстом молча.
    # Здесь же правило одно: теги дописываются последней строкой шапки.
    lines = text.split("\n")
    by_start = {r.start: r for r in records}
    out: list[str] = []
    i = 0
    while i < len(lines):
        out.append(lines[i])
        rec = by_start.get(i)
        if rec is None:
            i += 1
            continue
        # Пропускаем пустую строку и собираем шапку до следующей пустой.
        j = i + 1
        while j < len(lines) and lines[j].strip() == "":
            out.append(lines[j])
            j += 1
        head: list[str] = []
        while j < len(lines) and lines[j].strip() != "":
            head.append(lines[j])
            j += 1
        if head and head[0].startswith("**Область:**"):
            # Прежняя строка тегов, если запуск повторный, заменяется, а не
            # дублируется: указатель обязан быть перестроим на месте. Фильтр
            # берёт ТОЛЬКО шапку текущей записи — по всему выводу он снял бы
            # теги у всех предыдущих.
            head = [ln for ln in head if not ln.startswith("**Теги:**")]
            head.append(f"**Теги:** {', '.join(rec.subjects) or '—'}")
        out.extend(head)
        i = j
    text = "\n".join(out)

    index = build_index(records)
    if MARK_START in text:
        text = re.sub(
            re.escape(MARK_START) + ".*?" + re.escape(MARK_END), index, text, flags=re.S
        )
    else:
        text = text.replace("\n## Указатель по областям", f"\n{index}\n\n## Указатель по областям", 1)
    return text


def linkify(text: str, known: set[str]) -> tuple[str, int, int]:
    """Проставить якоря у записей и превратить номера в ссылки.

    ★Порядок работы: сперва всё РАЗБИРАЕТСЯ обратно (ссылки в голые номера,
    якоря удаляются), потом ставится заново. Так прогон идемпотентен по
    устройству, а не по аккуратности регулярного выражения: повторный запуск не
    может дать `[[Р-119](#r-119)](#r-119)`, потому что вложенного случая просто
    не возникает.

    Ссылкой становится номер ВЕЗДЕ, кроме заголовка самой записи: внутри записи
    ссылка на соседнюю нужна ровно так же, как в указателе, — «отменяет Р-061»
    без перехода заставляет читателя искать вручную по шести тысячам строк.

    Номер, для которого записи нет, ссылкой НЕ становится. Такой случай — ошибка
    (в реестре сейчас таких нет, но появиться они могут), и превращать её в
    ссылку в никуда значит прятать.
    """
    text = LINKED_RE.sub(r"\1", text)
    text = ANCHOR_RE.sub("", text)

    anchors = 0
    links = 0
    out: list[str] = []
    for line in text.split("\n"):
        m = re.match(r"^### (Р-(\d+))\. ", line)
        if m:
            out.append(ANCHOR.format(n=m.group(2)))
            anchors += 1
            out.append(line)  # заголовок не трогаем: ссылка сама на себя не нужна
            continue

        def repl(mo: re.Match[str]) -> str:
            nonlocal links
            if mo.group(1) not in known:
                return mo.group(1)
            links += 1
            return f"[{mo.group(1)}](#r-{mo.group(2)})"

        out.append(NUMBER_RE.sub(repl, line))
    return "\n".join(out), anchors, links


def main() -> int:
    text = REGISTRY.read_text(encoding="utf-8")
    records = parse(text)
    assign(records)
    bad = report(records)

    if "--apply" not in sys.argv:
        print("\n(отчёт; файл не тронут — запустите с --apply)")
        return 0

    shutil.copy(REGISTRY, REGISTRY.with_suffix(".md.bak"))
    new = apply(text, records)
    new, anchors, links = linkify(new, {r.number for r in records})
    REGISTRY.write_text(new, encoding="utf-8")
    print(f"\nякорей: {anchors} · ссылок на записи: {links}")
    print(f"записано; резервная копия — {REGISTRY.with_suffix('.md.bak').name}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
