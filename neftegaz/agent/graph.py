"""The agent itself: a LangGraph state machine over explicit steps.

The graph is written out as named nodes rather than handed to a generic
tool-calling loop, because requirement 2.4 prescribes a *specific* source
priority — reports first, web as a supplement — and a prescribed policy should
be visible in the code that implements it. Someone auditing this system can
read the routing here and check it against the requirement, line by line, which
is not possible when the policy lives inside a model's discretion.

    route ─┬─► out_of_scope ──────────────► END
           ├─► forecast ─────► answer ────► END
           └─► retrieve ─► (web?) ─► answer ─► END
"""

from __future__ import annotations

import re
from typing import Any, Literal, TypedDict

from neftegaz.agent import prompts
from neftegaz.agent.llm import ask
from neftegaz.config import settings

__all__ = ["AgentState", "build_graph", "answer_question"]


class AgentState(TypedDict, total=False):
    """What flows through the graph.

    Every intermediate is kept, not just the final text: the UI shows which
    sources were consulted, and the demo transcripts have to prove that the
    source-priority rule actually ran.
    """

    question: str
    route: str
    report_hits: list[Any]
    web_hits: list[Any]
    forecast_text: str
    used_web: bool
    used_reports: bool
    answer: str


# ── helpers ────────────────────────────────────────────────────────────────

# Words that mean "the corpus may be stale, go and look at the web". Reports
# are quarterly or monthly; anything asking about right now cannot be answered
# from them however good the retrieval score is.
FRESHNESS_MARKERS = (
    "сейчас", "сегодня", "вчера", "текущ", "актуальн", "последн", "свеж",
    "новост", "на данный момент", "котировк", "заявил", "объявил",
)

MONTHS_TO_DAYS = 30
YEARS_TO_DAYS = 365
WEEKS_TO_DAYS = 7


def _needs_fresh_data(question: str) -> bool:
    lowered = question.lower()
    return any(marker in lowered for marker in FRESHNESS_MARKERS)


def parse_horizon_days(question: str, default: int = 90) -> int:
    """Pull a forecast horizon out of the question.

    Regex rather than another LLM call: the pattern space is tiny and closed,
    and a deterministic parser cannot hallucinate a horizon of 10 000 days.
    """
    lowered = question.lower()
    patterns = [
        (r"(\d+)\s*(?:кв|квартал)", MONTHS_TO_DAYS * 3),
        (r"(\d+)\s*(?:мес|month)", MONTHS_TO_DAYS),
        (r"(\d+)\s*(?:год|лет|года|year)", YEARS_TO_DAYS),
        (r"(\d+)\s*(?:недел|week)", WEEKS_TO_DAYS),
        (r"(\d+)\s*(?:дн|день|дней|day)", 1),
    ]
    for pattern, multiplier in patterns:
        found = re.search(pattern, lowered)
        if found:
            value = int(found.group(1)) * multiplier
            # Clamp: a forecast further out than the history is arithmetic, not
            # analysis, and the band would be wider than the price.
            return max(1, min(value, 730))
    return default


def parse_supply_change(question: str) -> float:
    """Extract a supply scenario in million barrels per day, if stated.

    Returns 0.0 when no scenario is present. Negative means a cut.
    """
    lowered = question.lower()
    found = re.search(r"(\d+[.,]?\d*)\s*млн\s*барр", lowered)
    if not found:
        return 0.0
    magnitude = float(found.group(1).replace(",", "."))
    cut_words = ("сокращ", "снижен", "уменьш", "срез", "cut", "reduc")
    is_cut = any(word in lowered for word in cut_words)
    return -magnitude if is_cut else magnitude


def _format_report_context(hits: list[Any]) -> str:
    """Render retrieved chunks with the metadata the citation format needs."""
    blocks = []
    for hit in hits:
        header = (
            f"[фрагмент: {hit.source_name}, {hit.date}, с. {hit.page}"
            + (f"–{hit.page_end}" if hit.page_end != hit.page else "")
            + f", близость {hit.score:.3f}]"
        )
        blocks.append(f"{header}\n{hit.text}")
    return "\n\n".join(blocks)


def _format_web_context(hits: list[Any]) -> str:
    blocks = []
    for hit in hits:
        mark = " (отраслевой/агентский источник)" if hit.preferred else ""
        blocks.append(f"[веб: {hit.title} — {hit.domain}{mark}]\n{hit.snippet}\n{hit.url}")
    return "\n\n".join(blocks)


# ── nodes ──────────────────────────────────────────────────────────────────


def node_route(state: AgentState) -> AgentState:
    """Classify the question into one of the graph's three branches."""
    question = state["question"]

    # A forecast request is recognised structurally before asking the model:
    # these phrasings are unambiguous, and skipping a round trip on them makes
    # the common case both faster and immune to a classifier wobble.
    lowered = question.lower()
    if any(word in lowered for word in ("спрогнозируй", "прогноз цен", "оцени диапазон", "спрогнозировать")):
        return {"route": "forecast"}

    try:
        verdict = ask("Ты классификатор запросов. Отвечай одним словом.",
                      prompts.ROUTER_PROMPT.format(question=question))
    except Exception:  # noqa: BLE001 - classifier failure must not kill the answer
        # Default to the industry branch: answering a cooking question with oil
        # analysis is embarrassing, but refusing a legitimate industry question
        # because the classifier timed out is a broken product.
        return {"route": "industry"}

    return {"route": parse_route(verdict)}


ROUTE_NAMES = ("forecast", "industry", "other")


def parse_route(verdict: str, default: str = "industry") -> str:
    """Extract the chosen category from a classifier reply.

    Two rules, both learned the hard way:

    * Match whole words, not substrings. "Не forecasting" contains "forecast",
      and so does a reasoning model listing the options it rejected.
    * Take the LAST match, not the first. A model that deliberates before
      answering mentions candidates in the order they appear in the prompt;
      its actual choice is whatever it says last. Taking the first match makes
      the answer depend on the order of the option list rather than on the
      model's decision — which is precisely the defect this replaced: every
      question routed to "forecast" because that word led the search list and
      appeared in every monologue.

    The reasoning block is stripped here as well as in `ask`, deliberately.
    The parser must not depend on another layer having cleaned up first — and
    a reply truncated mid-monologue keeps its <think> tag unclosed, so the
    words inside it survive any regex that only removes matched pairs.
    """
    from neftegaz.agent.llm import strip_reasoning

    found = re.findall(r"\b(forecast|industry|other)\b", strip_reasoning(verdict).lower())
    return found[-1] if found else default


def node_retrieve(state: AgentState) -> AgentState:
    """Search the report corpus. This always runs first for industry questions."""
    from neftegaz.rag.store import get_store

    try:
        hits = get_store().search(state["question"])
    except Exception:  # noqa: BLE001 - an unbuilt index is a normal first-run state
        hits = []
    return {"report_hits": hits, "used_reports": bool(hits)}


def node_web(state: AgentState) -> AgentState:
    """Search the web. Reached only when reports are insufficient or stale."""
    from neftegaz.tools.web import search_web

    hits = search_web(state["question"])
    return {"web_hits": hits, "used_web": bool(hits)}


def node_forecast(state: AgentState) -> AgentState:
    """Run the calculation module and hand its output to the answering step."""
    from neftegaz.tools.forecast_tool import run_forecast

    question = state["question"]
    try:
        report = run_forecast(
            horizon_days=parse_horizon_days(question),
            supply_change_mb_d=parse_supply_change(question),
        )
        return {"forecast_text": report.as_text()}
    except FileNotFoundError as exc:
        return {"forecast_text": f"Расчётный модуль недоступен: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"forecast_text": f"Расчёт не выполнен: {exc}"}


def node_answer(state: AgentState) -> AgentState:
    """Compose the final answer from whatever the branches gathered."""
    question = state["question"]
    report_context = _format_report_context(state.get("report_hits") or [])
    web_context = _format_web_context(state.get("web_hits") or [])

    # The calculation is a first-class source — reproducible, unlike the web —
    # but it is a source of a *different kind*, so it gets its own section and
    # its own citation format rather than being folded into the report context.
    forecast_text = state.get("forecast_text", "")

    try:
        answer = ask(
            prompts.SYSTEM_PROMPT,
            prompts.build_answer_prompt(question, report_context, web_context, forecast_text),
        )
    except Exception as exc:  # noqa: BLE001
        # Degrade to the raw material rather than losing the work: a forecast
        # the user can read beats an error page.
        fallback = forecast_text or report_context or web_context
        answer = (
            f"Языковая модель недоступна ({exc}).\n\n"
            f"Собранные данные:\n\n{fallback}" if fallback
            else f"Языковая модель недоступна: {exc}"
        )
    return {"answer": answer}


def node_out_of_scope(state: AgentState) -> AgentState:
    return {"answer": prompts.OUT_OF_SCOPE_REPLY, "used_reports": False, "used_web": False}


# ── edges ──────────────────────────────────────────────────────────────────


def _after_route(state: AgentState) -> Literal["forecast", "retrieve", "out_of_scope"]:
    route = state.get("route", "industry")
    if route == "forecast":
        return "forecast"
    if route == "other":
        return "out_of_scope"
    return "retrieve"


def _after_retrieve(state: AgentState) -> Literal["web", "answer"]:
    """The source-priority rule from requirement 2.4, in one place.

    Web search runs when either the corpus came up short, or the question is
    about the present — reports are periodic and cannot answer "what is the
    price today" however well they match.
    """
    hits = state.get("report_hits") or []
    if len(hits) < max(1, settings.top_k // 2):
        return "web"
    if _needs_fresh_data(state["question"]):
        return "web"
    return "answer"


def build_graph():
    """Compile the state machine."""
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AgentState)
    graph.add_node("route", node_route)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("web", node_web)
    graph.add_node("forecast", node_forecast)
    graph.add_node("answer", node_answer)
    graph.add_node("out_of_scope", node_out_of_scope)

    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", _after_route,
                                {"forecast": "forecast", "retrieve": "retrieve", "out_of_scope": "out_of_scope"})
    graph.add_conditional_edges("retrieve", _after_retrieve, {"web": "web", "answer": "answer"})
    graph.add_edge("web", "answer")
    # ★Расчёт ведёт в поиск по корпусу, а не прямо в ответ. Требование 2.4
    # задаёт ПРИОРИТЕТ источников, а не выбор одного из них: собственная
    # модель дополняет отчёты. Прямое ребро в ответ стоило конкретного
    # дефекта — вопрос «какой прогноз EIA по Brent на 2027 год» уходил в
    # ветку прогноза и получал «в базе отчётов ничего не найдено», хотя поиск
    # по тому же вопросу возвращает фрагменты с близостью 0.72 при пороге
    # 0.55 и нужная таблица STEO в корпусе есть.
    graph.add_edge("forecast", "retrieve")
    graph.add_edge("answer", END)
    graph.add_edge("out_of_scope", END)
    return graph.compile()


_COMPILED = None


def answer_question(question: str) -> AgentState:
    """Run one question through the agent and return the full final state."""
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_graph()
    initial: AgentState = {"question": question, "used_web": False, "used_reports": False}
    return _COMPILED.invoke(initial)
