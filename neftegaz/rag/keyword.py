"""Поиск по словам (BM25) и русско-английский мостик к нему.

ЗАЧЕМ ОН НУЖЕН РЯДОМ С ВЕКТОРНЫМ. Замерено на нашем корпусе: строка таблицы
«United States ......... 13.28 13.51 13.78» по чистому косинусу стоит на 99-м
месте (0.64 против 0.752 у лидера) и в выдачу не попадает никогда. Из слов у неё
есть только «United States» и заголовок таблицы — эмбеддеру не за что зацепиться,
и никакая настройка ранжирования этого не меняет. Тот же фрагмент поиск по словам
ставит на 9-11-е место. Два способа ошибаются по-разному, и потому их слияние
находит то, чего не находит ни один.

★МОСТИК ЧЕРЕЗ ЯЗЫК — УСЛОВИЕ, БЕЗ КОТОРОГО ЗАТЕЯ МОЛЧА ПРОВАЛИВАЕТСЯ. Корпус
английский, вопросы русские: у запроса «добыча нефти в США» с английским
фрагментом НЕТ НИ ОДНОГО общего слова, и BM25 честно вернёт пустоту. Отказ был бы
тихим — выдача просто осталась бы прежней, и это выглядело бы как «гибрид не
помог». Поэтому словарь предметной области здесь не украшение, а несущая часть.

Словарь выбран вместо перевода моделью намеренно: словарь предметной области
мал, закрыт и проверяется тестами, а вызов модели добавил бы задержку к каждому
запросу и ещё один отказ, который надо обрабатывать.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

__all__ = ["GLOSSARY", "expand_query", "tokenize", "BM25Index"]

# ── русско-английский мостик ───────────────────────────────────────────────
# Слева — НАЧАЛА русских слов (не целые слова): русский склоняется, и «нефть /
# нефти / нефтью» должны попасть одинаково. Справа — то, как это называется в
# отчётах EIA. Сопоставление по началу слова грубее морфологического разбора и
# ровно поэтому надёжнее: оно не знает исключений, но и не ошибается на них.
GLOSSARY: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("добыч", "производств"), ("production",)),
    (("нефт", "сыр"), ("oil", "crude", "petroleum")),
    (("запас", "резерв", "хранилищ"), ("inventories", "reserves", "stocks", "storage")),
    (("сша", "америк", "штат"), ("united", "states")),
    (("цен", "стоимост", "котиров"), ("price", "prices", "spot")),
    (("спрос", "потреблен"), ("consumption", "demand")),
    (("предложен", "поставк"), ("supply",)),
    (("газ",), ("gas", "natural")),
    (("опек",), ("opec",)),
    (("бензин",), ("gasoline",)),
    (("дизел", "мазут"), ("distillate", "diesel")),
    (("переработк", "нпз", "рафинир"), ("refinery", "refining", "refiner")),
    (("экспорт",), ("exports", "export")),
    (("импорт",), ("imports", "import")),
    (("брент",), ("brent",)),
    (("прогноз",), ("forecast", "forecasts")),
    (("миров", "глобальн"), ("world", "global")),
    (("баланс",), ("balance", "balances")),
    (("электро", "электрич"), ("electricity", "generation"),),
    (("уголь", "угл"), ("coal",)),
    (("сланц",), ("shale", "tight")),
    (("квот", "сокращен"), ("quota", "cuts", "curtailment")),
)

_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]{2,}")


def tokenize(text: str) -> list[str]:
    """Слова в нижнем регистре. Числа намеренно отброшены.

    Числа в этом корпусе — плохие поисковые слова: «13.28» встречается в десятках
    таблиц про совершенно разное, и совпадение по нему означает не близость темы,
    а случайность. Искать по словам и находить по числам — разные вещи.
    """
    return [word.lower() for word in _WORD.findall(text)]


def expand_query(question: str) -> list[str]:
    """Слова запроса, переведённые в лексику корпуса.

    Латиница проходит насквозь: вопрос может быть задан и по-английски, и
    смешанно («прогноз Brent»), и терять уже подходящее слово было бы глупо.
    """
    words = tokenize(question)
    expanded: list[str] = []
    for word in words:
        if word.isascii():
            expanded.append(word)
            continue
        for prefixes, english in GLOSSARY:
            if word.startswith(prefixes):
                expanded.extend(english)
    # Порядок сохраняется, повторы убираются: повтор слова в запросе BM25 не
    # усиливает — он лишь удваивает вклад одного и того же признака.
    seen: set[str] = set()
    return [w for w in expanded if not (w in seen or seen.add(w))]


# ── BM25 ───────────────────────────────────────────────────────────────────

K1 = 1.5   # насыщение по частоте слова в документе
B = 0.75   # насколько учитывать длину документа


@dataclass
class BM25Index:
    """Классический BM25 поверх заранее собранного списка документов.

    Своя реализация вместо библиотеки — потому что нужно ровно это, а всякая
    зависимость в поставке заказчику должна себя оправдывать. Корпус (около
    девяти с половиной тысяч фрагментов) целиком помещается в память, индекс
    строится за секунды и живёт вместе с процессом.
    """

    documents: list[list[str]] = field(default_factory=list)
    frequencies: list[Counter] = field(default_factory=list)
    document_frequency: Counter = field(default_factory=Counter)
    lengths: list[int] = field(default_factory=list)
    average_length: float = 0.0

    def add(self, text: str) -> None:
        words = tokenize(text)
        counts = Counter(words)
        self.documents.append(words)
        self.frequencies.append(counts)
        self.lengths.append(len(words))
        for word in counts:
            self.document_frequency[word] += 1

    def finalise(self) -> None:
        total = len(self.lengths)
        # Единица, а не ноль, для пустого корпуса: средняя длина стоит в
        # знаменателе, и «нет документов» не должно превращаться в деление на
        # ноль где-то глубже, где причину уже не видно.
        self.average_length = (sum(self.lengths) / total) if total else 1.0

    def idf(self, word: str) -> float:
        total = len(self.documents)
        appearances = self.document_frequency.get(word, 0)
        # Сглаженная форма: слово, встречающееся почти везде, получает вес около
        # нуля, но не отрицательный — иначе частое слово наказывало бы документ.
        return math.log(1 + (total - appearances + 0.5) / (appearances + 0.5))

    def rank(self, words: list[str], limit: int) -> list[tuple[int, float]]:
        """Лучшие документы: список (номер документа, вес), по убыванию веса."""
        if not words or not self.documents:
            return []
        weights = {word: self.idf(word) for word in set(words)}
        scored: list[tuple[int, float]] = []
        for index, counts in enumerate(self.frequencies):
            total = 0.0
            for word in words:
                frequency = counts.get(word, 0)
                if not frequency:
                    continue
                norm = 1 - B + B * self.lengths[index] / self.average_length
                total += weights[word] * frequency * (K1 + 1) / (frequency + K1 * norm)
            if total > 0:
                scored.append((index, total))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]
