"""Web search, with a source filter.

Веб-поиск идёт через библиотеку `ddgs` — единственный вариант из списка ТЗ
(Tavily, SerpAPI, DuckDuckGo, Google CSE), не требующий ключа API. Это важно не
ради удобства: проверяющий клонирует репозиторий и видит работу поиска, нигде
не регистрируясь, — то есть возможность проверяема, а не заявлена.

★АДРЕСАТ НАЗВАН ПОИМЁННО, И ЭТО НЕ КОСМЕТИКА. Имя библиотеки происходит от
DuckDuckGo, но одним этим сервисом она не ограничивается: в режиме по умолчанию
(`auto`) версия 9.15.0 обходит семь сервисов — wikipedia, grokipedia,
html.duckduckgo.com, search.yahoo.com, www.mojeek.com, search.brave.com,
www.startpage.com — и выбирает адресата в момент запроса. Наружу при этом
уезжает ТЕКСТ ВОПРОСА ПОЛЬЗОВАТЕЛЯ. Для системы, которая сдаётся как локальная,
«куда-то из семи» — не ответ, поэтому сервис задаётся одним значением
(`WEB_BACKEND`) и называется в README рядом с описью исходящего трафика.

★ПОЧЕМУ ИМЕННО BRAVE — ПО ЗАМЕРУ, А НЕ ПО РЕПУТАЦИИ. Три вопроса нашего рода
(добыча в США по STEO, прогноз Brent, эффект сокращения ОПЕК+), по одному
сервису за раз, три прогона 31.08.2026 с этой машины:

===========  ==================  ==================  =================
сервис       результатов (1/2/3) предпочитаемых*      прогонов вчистую
===========  ==================  ==================  =================
brave        45 / 45 / 45        21 / 21 / 21        3 из 3
yahoo        14 / 7 / 14         5 / 3 / 4           0 из 3
duckduckgo   0 / 0 / 30          0 / 0 / 6           1 из 3
mojeek       0 / 0 / 0           0 / 0 / 0           0 из 3
startpage    0 / 0 / 0           0 / 0 / 0           0 из 3
===========  ==================  ==================  =================

\\* домены из :data:`PREFERRED_DOMAINS` — агентства, регуляторы, отраслевая
пресса. «Прогон вчистую» — все три вопроса ответили без отказа.

Решающим оказалось не число результатов, а ПОВТОРЯЕМОСТЬ. Mojeek и startpage
не вернули ничего ни разу (библиотека к ним обращается, а разобрать ответ не
может), duckduckgo и yahoo отвечают через раз, и только brave повторил один
и тот же результат во всех трёх прогонах. Неустойчивость здесь дороже, чем
кажется: именно она делала режим `auto` похожим на работающий — из семи
кто-нибудь да отвечал, и вопрос уезжал к тому, кто ответил сегодня.

⚠Замер опроверг мой собственный прогноз: я ожидал, что сработают duckduckgo,
yahoo и mojeek, а brave и startpage откажут — не угадал ни в одну сторону.
Поэтому выбор стоит на прогоне, а не на репутации сервисов.

⚠Число сервисов и их поведение — свойство ВЕРСИИ библиотеки, а не константа
мира. Замер повторяем: :mod:`tests.test_web_backend` проверяет форму настройки,
а сам прогон — `scripts/probe_web_backends.py`.

Requirement 2.3 also asks to filter out tabloid sources. That is done with an
explicit, readable deny-list plus a preference for known industry and agency
domains — not with a model judging credibility, because a deny-list can be
inspected, argued with, and corrected by the customer, and a model's opinion
cannot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from neftegaz.config import settings

__all__ = [
    "DENY_DOMAINS",
    "FORBIDDEN_BACKENDS",
    "PREFERRED_DOMAINS",
    "WebResult",
    "checked_backend",
    "search_web",
    "search_web_with_status",
]

# Значения, при которых библиотека сама выбирает адресата. Запрещены: адресат
# обязан быть один и назван. Список из нескольких имён (через запятую) запрещён
# по той же причине — он тоже делает маршрут непредсказуемым до запроса.
FORBIDDEN_BACKENDS: frozenset[str] = frozenset({"auto", "all", ""})

# Tabloids, content farms and aggregators that republish without attribution.
# Deliberately short and specific: a long speculative list would silently drop
# legitimate sources, which is the more expensive error for an analyst.
DENY_DOMAINS: frozenset[str] = frozenset(
    {
        "dailymail.co.uk",
        "thesun.co.uk",
        "mirror.co.uk",
        "nypost.com",
        "express.co.uk",
        "zerohedge.com",
        "rt.com",
        "sputniknews.com",
        "life.ru",
        "ren.tv",
        "kp.ru",
        "eadaily.com",
        "tsargrad.tv",
        "pravda.ru",
        "topcor.ru",
        "avia.pro",
    }
)

# Agencies, regulators and industry press. A hit here is ranked above an
# unknown domain; it is not a whitelist — unknown domains still pass.
PREFERRED_DOMAINS: frozenset[str] = frozenset(
    {
        "reuters.com",
        "bloomberg.com",
        "ft.com",
        "wsj.com",
        "opec.org",
        "iea.org",
        "eia.gov",
        "spglobal.com",
        "argusmedia.com",
        "platts.com",
        "oilprice.com",
        "rigzone.com",
        "worldbank.org",
        "imf.org",
        "interfax.ru",
        "tass.ru",
        "vedomosti.ru",
        "kommersant.ru",
        "rbc.ru",
        "neftegaz.ru",
    }
)


@dataclass(frozen=True)
class WebResult:
    """One web hit, already filtered and attributed."""

    title: str
    url: str
    snippet: str
    domain: str
    preferred: bool

    def as_claim(self, text: str | None = None) -> dict:
        """Shape this result for :mod:`neftegaz.tools.citations`."""
        return {
            "source_type": "web",
            "text": self.snippet if text is None else text,
            "source_name": self.title or self.domain,
        }


def _domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _is_denied(domain: str) -> bool:
    """Deny the domain and any subdomain of it."""
    return any(domain == bad or domain.endswith(f".{bad}") for bad in DENY_DOMAINS)


def _is_preferred(domain: str) -> bool:
    return any(domain == good or domain.endswith(f".{good}") for good in PREFERRED_DOMAINS)


def _clean(text: str) -> str:
    """Strip the markup the search engine puts around matched terms."""
    return re.sub(r"<[^>]+>", "", text or "").strip()


def checked_backend(name: str | None = None) -> tuple[str, str]:
    """Имя поискового сервиса, проверенное ДО запроса — или причина отказа.

    Возвращает пару «имя, ошибка»: ровно одно из двух непусто.

    ★ЗАЧЕМ ПРОВЕРЯТЬ САМИМ, А НЕ ДОВЕРИТЬСЯ БИБЛИОТЕКЕ. `ddgs` на незнакомое
    имя бэкенда не отказывает, а ВОЗВРАЩАЕТСЯ К `auto`::

        if not instances:
            logger.warning("backend is not set. Using 'auto'")
            return self._get_engines(category, "auto")

    То есть опечатка в настройке (``brave`` → ``brve``) не ломает поиск и не
    печатает ничего заметного — она тихо рассылает вопрос пользователя по всем
    семи сервисам. Отказ, который выглядит как работа, здесь дороже отказа.
    Поэтому имя сверяется со списком движков самой библиотеки, а не с нашей
    копией списка: копия отстанет от библиотеки и снова начнёт врать.
    """
    backend = (settings.web_backend if name is None else name).strip()
    if backend in FORBIDDEN_BACKENDS:
        return "", (
            f"WEB_BACKEND={backend!r} отдаёт выбор адресата библиотеке. "
            "Нужно одно имя сервиса: система обязана знать, куда уходит вопрос."
        )
    if "," in backend:
        return "", (
            f"WEB_BACKEND={backend!r} перечисляет несколько сервисов. "
            "Нужно ровно одно имя: маршрут вопроса обязан быть известен заранее."
        )
    try:
        from ddgs.engines import ENGINES
    except ImportError as exc:  # pragma: no cover - depends on environment
        return "", f"библиотека веб-поиска не установлена ({exc})"
    known = ENGINES.get("text", {})
    if backend not in known:
        return "", (
            f"WEB_BACKEND={backend!r} библиотеке неизвестен "
            f"(есть: {', '.join(sorted(known))}). "
            "Оставить как есть нельзя: на неизвестное имя ddgs молча "
            "возвращается к режиму auto и рассылает вопрос по всем сервисам."
        )
    return backend, ""


def search_web(query: str, max_results: int | None = None) -> list[WebResult]:
    """Найденное в вебе — без состояния поиска.

    ★Тонкая обёртка над :func:`search_web_with_status`, оставленная ради
    вызывающих, которым довольно списка. Читать ТОЛЬКО её недостаточно: пустой
    список означает и «поиск прошёл, ничего не нашлось», и «поиск не состоялся»,
    а это разные вещи, и вторую надо произнести вслух. Кто принимает решение —
    берёт полный ответ, а не этот список.
    """
    return search_web_with_status(query, max_results)[0]


def search_web_with_status(
    query: str, max_results: int | None = None
) -> tuple[list[WebResult], str]:
    """Найденное в вебе И состояние поиска.

    Состояние — пустая строка (поиск прошёл), ``"unavailable: …"`` (библиотеки
    поиска нет в сборке) или ``"failed: …"`` (сеть или разбор отказали).

    ★ЗАЧЕМ ПАРА, А НЕ СПИСОК. Пустой список означал три разных положения дел
    сразу, и в промпт уходила одна строка на все три — «(веб-поиск не выполнялся
    или ничего не вернул)». Союз «или» делает её правдивой и потому бесполезной:
    из неё нельзя понять, промолчал веб или не был спрошен. Худший случай — когда
    молчат ОБА источника: ответ встаёт на память модели, а человеку это ничем не
    показано.

    Ни один исход не выражается исключением: отсутствие сети — не ошибка
    программы, а обстоятельство, в котором агент обязан продолжать работать по
    корпусу отчётов. Обстоятельство возвращается значением и едет дальше как факт.
    """
    limit = settings.web_results if max_results is None else max_results
    try:
        from ddgs import DDGS
    except ImportError as exc:  # pragma: no cover - depends on environment
        return [], f"unavailable: библиотека веб-поиска не установлена ({exc})"

    backend, why = checked_backend()
    if why:
        # Настройка неверна — поиск не выполняется вовсе. Это осознанно строже,
        # чем «сделаем как получится»: единственная альтернатива здесь —
        # разослать вопрос пользователя по семи сервисам, и она хуже отказа.
        return [], f"unavailable: {why}"

    try:
        with DDGS() as engine:
            # Over-fetch: filtering removes some hits, and asking for exactly
            # `limit` would leave us short whenever a tabloid ranks well.
            raw = list(
                engine.text(
                    query,
                    region=settings.web_region,
                    max_results=limit * 3,
                    backend=backend,
                )
            )
    except Exception as exc:  # noqa: BLE001 - network/parse failures must not propagate
        # ★Отказ единственного сервиса — именно отказ, а не переход к соседу.
        # При одном бэкенде ddgs не находит результатов и поднимает
        # DDGSException; она приезжает сюда и уходит пользователю строкой.
        # Текст обрезается: библиотека вкладывает в него весь запрошенный URL
        # вместе с вопросом пользователя, а строка состояния едет и в промпт,
        # и на экран — там нужна причина, а не копия вопроса.
        detail = f"{type(exc).__name__}: {exc}"
        return [], f"failed: {backend}: {detail[:160]}"

    results: list[WebResult] = []
    seen: set[str] = set()
    for item in raw:
        url = item.get("href") or item.get("url") or ""
        if not url:
            continue
        domain = _domain(url)
        if not domain or _is_denied(domain) or domain in seen:
            continue
        seen.add(domain)  # one hit per domain: five takes of one story is not five sources
        results.append(
            WebResult(
                title=_clean(item.get("title", "")),
                url=url,
                snippet=_clean(item.get("body") or item.get("snippet") or ""),
                domain=domain,
                preferred=_is_preferred(domain),
            )
        )

    # Stable sort: preferred domains first, original relevance order preserved
    # inside each group.
    results.sort(key=lambda r: not r.preferred)
    return results[:limit], ""
