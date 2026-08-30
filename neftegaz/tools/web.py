"""Web search, with a source filter.

DuckDuckGo is the one option from the assignment's list (Tavily, SerpAPI,
DuckDuckGo, Google CSE) that needs no API key. That matters beyond convenience:
a reviewer can clone the repository and watch web search work without
registering anywhere, so this capability is verifiable rather than merely
claimed.

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
    "PREFERRED_DOMAINS",
    "WebResult",
    "search_web",
    "search_web_with_status",
]

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
    """Strip the markup DuckDuckGo puts around matched terms."""
    return re.sub(r"<[^>]+>", "", text or "").strip()


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

    try:
        with DDGS() as engine:
            # Over-fetch: filtering removes some hits, and asking for exactly
            # `limit` would leave us short whenever a tabloid ranks well.
            raw = list(engine.text(query, region=settings.web_region, max_results=limit * 3))
    except Exception as exc:  # noqa: BLE001 - network/parse failures must not propagate
        return [], f"failed: {exc}"

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
