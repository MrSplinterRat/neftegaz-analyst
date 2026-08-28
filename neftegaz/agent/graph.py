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
from typing import Annotated, Any, Literal, TypedDict

from neftegaz.agent import prompts
from neftegaz.agent.llm import ask
from neftegaz.config import settings

__all__ = [
    "AgentState",
    "build_graph",
    "answer_question",
    "merge_history",
    "format_history",
    "new_thread_id",
    "parse_horizon_days",
    "requested_horizon_days",
    "parse_supply_change",
]


def _cap_turn(turn: dict) -> dict:
    """Подрезать ответ одного хода до потолка, заданного настройкой."""
    cap = settings.history_turn_cap_chars
    answer = turn.get("answer", "")
    if cap <= 0 or len(answer) <= cap:
        return turn
    return {**turn, "answer": answer[:cap] + TRUNCATION_MARK}


def merge_history(old: list[dict], new: list[dict]) -> list[dict]:
    """Склеить ходы разговора и обрезать СТАРЫЕ, если бюджет исчерпан.

    Это reducer состояния LangGraph: он вызывается на каждом обновлении поля
    ``history``, поэтому обрезание происходит ПРИ ЗАПИСИ, а не только при
    сборке промпта. Разница существенна для файлового хранилища: иначе база
    разговоров растёт без границы, а в промпт всё равно попадает хвост.

    ★Старые ходы отбрасываются ЦЕЛИКОМ, без пересказа. Суммаризация означала
    бы ещё один вызов модели на каждом шаге — задержка, деньги и недетерминизм
    там, где его быть не должно: два одинаковых разговора обязаны давать
    одинаковый промпт.

    Отбрасывается голова, а не хвост: свежие ходы нужнее для разрешения
    «а на пять лет?», чем начало разговора.
    """
    budget = settings.history_budget_chars
    if budget <= 0:
        return []

    # ★Ход подрезается ПРИ ЗАПИСИ, а не только при сборке промпта. Иначе один
    # реальный ответ (три-пять тысяч знаков) съедает весь бюджет целиком, и
    # история схлопывается до единственного последнего хода — то есть память
    # выглядит работающей, а на деле не помнит ничего. Ровно это и случилось
    # на первом же сквозном прогоне.
    merged = [_cap_turn(turn) for turn in [*(old or []), *(new or [])]]

    kept: list[dict] = []
    spent = 0
    for turn in reversed(merged):
        cost = len(turn.get("question", "")) + len(turn.get("answer", ""))
        if spent + cost > budget and kept:
            break
        kept.append(turn)
        spent += cost
    kept.reverse()
    return kept


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
    # Горизонт последнего расчёта в этом разговоре. Обычное поле состояния:
    # узел, который его не вернул, оставляет прежнее значение, а чекпоинтер
    # переносит его в следующий вопрос.
    last_horizon_days: int
    # Ходы разговора. Аннотация задаёт reducer: LangGraph не заменяет поле
    # новым значением, а прогоняет старое и новое через merge_history.
    history: Annotated[list[dict], merge_history]


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


# Числительные словом, в формах, которые встречаются перед единицей времени.
# ★«Полтора» намеренно НЕТ: округлить его до года значило бы ответить на другой
# вопрос, а не признать, что вопрос не разобран. Умолчание честнее подмены.
NUMERALS = {
    "один": 1, "одного": 1, "одну": 1,
    "два": 2, "две": 2, "двух": 2, "пару": 2, "пары": 2,
    "три": 3, "трёх": 3, "трех": 3,
    "четыре": 4, "четырёх": 4, "четырех": 4,
    "пять": 5, "пяти": 5,
    "шесть": 6, "шести": 6,
    "семь": 7, "семи": 7,
    "восемь": 8, "восьми": 8,
    "девять": 9, "девяти": 9,
    "десять": 10, "десяти": 10,
    "двенадцать": 12, "двенадцати": 12,
}


def _numerals_to_digits(text: str) -> str:
    """Заменить числительные словом на цифры, чтобы дальше работал один разбор."""
    return re.sub(
        r"\b(" + "|".join(NUMERALS) + r")\b",
        lambda match: str(NUMERALS[match.group(1)]),
        text,
    )


def requested_horizon_days(question: str) -> int | None:
    """Pull the horizon the user ASKED FOR, in days, or None if none was named.

    Без приведения к допустимому диапазону — именно затем, чтобы вызывающий мог
    сравнить запрошенное с применённым и сказать вслух, что запрос урезан.

    Regex rather than another LLM call: the pattern space is tiny and closed,
    and a deterministic parser cannot hallucinate a horizon of 10 000 days.

    ★Числительные словом распознаются наравне с цифрами. В продолжении
    разговора человек почти не пишет цифру: «а на пять лет?», «а на год?» —
    и парсер, знающий только ``\\d+``, молча возвращал умолчание. Со сквозного
    прогона: вопрос «а на пять лет?» после прогноза на три месяца считался на
    те же 90 дней, а ответ выглядел так, будто система просто отказалась.
    """
    lowered = _numerals_to_digits(question.lower())
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
            return int(found.group(1)) * multiplier

    # Единица времени без числа означает одну: «а на год?», «на квартал».
    # Отдельным проходом, а не отдельной группой в регулярке выше, потому что
    # «на 3 года» не должно совпасть здесь раньше, чем там.
    bare = [
        (r"\bна\s+квартал", MONTHS_TO_DAYS * 3),
        (r"\bна\s+(?:год|году)\b", YEARS_TO_DAYS),
        (r"\bна\s+месяц\b", MONTHS_TO_DAYS),
        (r"\bна\s+недел", WEEKS_TO_DAYS),
    ]
    for pattern, value in bare:
        if re.search(pattern, lowered):
            return value
    return None


# Дальше истории прогноз — арифметика, а не анализ: полоса станет шире самой
# цены. ★Потолок ОБЪЯВЛЕН пользователю, а не применён молча: спросив пять лет и
# получив «горизонт 730 дн.», человек читает это как ответ на свой вопрос —
# и ошибается ровно в 2.5 раза.
HORIZON_CAP_DAYS = 730


def parse_horizon_days(question: str, default: int = 90) -> int:
    """Горизонт из вопроса, приведённый к допустимому диапазону."""
    requested = requested_horizon_days(question)
    return max(1, min(requested if requested is not None else default, HORIZON_CAP_DAYS))


def parse_supply_change(question: str) -> float:
    """Extract a supply scenario in million barrels per day, if stated.

    Returns 0.0 when no scenario is present. Negative means a cut.
    """
    lowered = question.lower()
    found = re.search(r"(\d+[.,]?\d*)\s*млн\s*барр", lowered)
    if not found:
        return 0.0
    magnitude = float(found.group(1).replace(",", "."))
    is_cut = any(word in lowered for word in CUT_WORDS)
    return -magnitude if is_cut else magnitude


# Корни слов, означающих сокращение предложения. ★Список корней, а не слов:
# «сокращение» и «сократит» — разные корни («сокращ» и «сократ»), и пропуск
# второго стоил конкретного дефекта. На вопросе «а если ОПЕК+ СОКРАТИТ добычу
# на 2 млн барр./сут?» расчёт посчитал сценарий УВЕЛИЧЕНИЯ предложения и выдал
# падение цены вместо роста — то есть ответ, противоположный вопросу.
#
# ★Ошибка знака здесь дороже, чем ошибка величины: диапазон «на сколько» читатель
# перепроверит, направление — примет на веру. Поэтому корни перечислены с запасом,
# и берутся полные, различающие формы: «пад» вошло бы в слово «западный».
CUT_WORDS = (
    "сокращ", "сократ", "снижен", "сниз", "уменьш", "срез", "урез",
    "упад", "паден", "выпаден", "потер", "выбыт", "останов", "сверн",
    "cut", "reduc", "decline", "drop",
)


# ★БЮДЖЕТ КОНТЕКСТА В ЗНАКАХ, А НЕ НАДЕЖДА НА ТО, ЧТО ВЛЕЗЕТ.
#
# Замер 26.08: на вопросе «сопоставь прогноз EIA с текущими котировками»
# сервер модели ответил 500 context_length_exceeded, и агент выродился в
# свалку сырых фрагментов вместо ответа. Пять фрагментов отчётов по ~1400
# знаков плюс пять веб-выдержек плюс задание перевалили за окно.
#
# ★ЛЕЧИТЬ УВЕЛИЧЕНИЕМ ОКНА НЕЛЬЗЯ: сколько его ни дай, число найденного
# умножается на длину найденного, и обе величины растут от настроек поиска.
# Единственная защита — потолок, который держит СОБИРАЮЩАЯ сторона.
#
# Урезается ХВОСТ, потому что попадания приходят по убыванию близости:
# первым выпадает наименее подходящее. Обрезанный фрагмент помечается явно —
# модель должна отличать «в отчёте этого нет» от «сюда не поместилось».
REPORT_BUDGET_CHARS = 7000
WEB_BUDGET_CHARS = 4000
FRAGMENT_CAP_CHARS = 1800
TRUNCATION_MARK = "\n[…фрагмент обрезан по бюджету контекста]"


def _fit(blocks: list[str], budget: int) -> list[str]:
    """Оставить столько блоков с начала, сколько влезает в бюджет знаков."""
    kept: list[str] = []
    spent = 0
    for block in blocks:
        if spent + len(block) > budget:
            break
        kept.append(block)
        spent += len(block)
    # Хотя бы один блок отдаётся всегда: пустой контекст превратил бы ответ по
    # источникам в ответ по памяти модели, а это худший исход, чем усечение.
    if not kept and blocks:
        kept = [blocks[0][:budget] + TRUNCATION_MARK]
    return kept


def _clip(text: str) -> str:
    return text if len(text) <= FRAGMENT_CAP_CHARS else text[:FRAGMENT_CAP_CHARS] + TRUNCATION_MARK


def format_history(turns: list[dict]) -> str:
    """Ходы разговора в виде, пригодном для промпта.

    Каждый ход подрезается по своему потолку: без него один длинный ответ
    съедает весь бюджет и вытесняет остальную историю — а вытесняется как раз
    то, к чему относится слово «а» в следующем вопросе.
    """
    cap = settings.history_turn_cap_chars
    lines = []
    for turn in turns or []:
        question = turn.get("question", "").strip()
        answer = turn.get("answer", "").strip()
        if cap > 0 and len(answer) > cap:
            answer = answer[:cap] + TRUNCATION_MARK
        lines.append(f"Вопрос: {question}\nОтвет: {answer}")
    return "\n\n".join(lines)


def _format_report_context(hits: list[Any]) -> str:
    """Render retrieved chunks with the metadata the citation format needs."""
    blocks = []
    for hit in hits:
        header = (
            f"[фрагмент: {hit.source_name}, {hit.date}, с. {hit.page}"
            + (f"–{hit.page_end}" if hit.page_end != hit.page else "")
            + f", близость {hit.score:.3f}]"
        )
        # ★Контекст таблицы идёт ОТДЕЛЬНОЙ строкой, а не приклеивается к тексту.
        # Строка таблицы без него — просто ряд чисел: модель не знает ни того,
        # что это за таблица, ни какому периоду принадлежит каждое значение, и
        # честно об этом пишет. Склеить его с text нельзя: на text стоит ссылка
        # на страницу, и он обязан совпадать с отчётом дословно.
        context = (getattr(hit, "context", "") or "").strip()
        if context:
            caption, _, columns = context.partition("\n")
            marks = [f"таблица: {caption}"] if caption else []
            if columns:
                marks.append(f"колонки по порядку: {columns}")
            blocks.append(f"{header}\n[{' | '.join(marks)}]\n{_clip(hit.text)}")
        else:
            blocks.append(f"{header}\n{_clip(hit.text)}")
    return "\n\n".join(_fit(blocks, REPORT_BUDGET_CHARS))


def _format_web_context(hits: list[Any]) -> str:
    blocks = []
    for hit in hits:
        mark = " (отраслевой/агентский источник)" if hit.preferred else ""
        blocks.append(f"[веб: {hit.title} — {hit.domain}{mark}]\n{_clip(hit.snippet)}\n{hit.url}")
    return "\n\n".join(_fit(blocks, WEB_BUDGET_CHARS))


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
        # История нужна классификатору не меньше, чем отвечающему: «а на пять
        # лет?» само по себе не относится ни к какой отрасли, и без
        # предыдущего хода уходит в отказ вместо расчёта.
        verdict = ask(
            "Ты классификатор запросов. Отвечай одним словом.",
            prompts.build_router_prompt(question, format_history(state.get("history") or [])),
        )
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
    # ★Горизонт НАСЛЕДУЕТСЯ из предыдущего расчёта в этом разговоре: после
    # «спрогнозируй Brent на 3 месяца» вопрос «а если ОПЕК+ сократит добычу на
    # 2 млн барр./сут?» должен считаться на те же 3 месяца, а не молча
    # съезжать на умолчание.
    #
    # ★А величина шока НЕ наследуется, и это решение, а не недоделка.
    # Наследуемый сценарий стал бы липким: пользователь, задав однажды
    # «при сокращении на 1.5», получал бы сценарные числа на все последующие
    # вопросы, не имея способа вернуться к базовому прогнозу иначе как начав
    # разговор заново. Горизонт нейтрален и нужен всегда; допущение о шоке
    # должно быть НАЗВАНО, иначе оно искажает ответ незаметно.
    horizon = parse_horizon_days(question, default=state.get("last_horizon_days", 90))
    requested = requested_horizon_days(question)
    try:
        report = run_forecast(
            horizon_days=horizon,
            supply_change_mb_d=parse_supply_change(question),
        )
        text = report.as_text()
        if requested is not None and requested > horizon:
            # ★Урезание запроса объявляется вслух. Спросив пять лет и получив
            # «горизонт 730 дн.», человек читает это как ответ на свой вопрос —
            # и ошибается в 2.5 раза. Молчаливый потолок хуже отказа.
            text += (
                f"\n\n★Запрошен горизонт {requested} дн., расчёт выполнен на "
                f"{horizon} дн. Дальше этого срока прогноз по одной лишь истории "
                f"цены перестаёт быть анализом: доверительный интервал становится "
                f"шире самой цены, а история короче горизонта."
            )
        return {"forecast_text": text, "last_horizon_days": horizon}
    except FileNotFoundError as exc:
        return {"forecast_text": f"Расчётный модуль недоступен: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"forecast_text": f"Расчёт не выполнен: {exc}"}


def node_answer(state: AgentState) -> AgentState:
    """Compose the final answer from whatever the branches gathered."""
    question = state["question"]
    report_context = _format_report_context(state.get("report_hits") or [])
    web_context = _format_web_context(state.get("web_hits") or [])
    history_context = format_history(state.get("history") or [])

    # The calculation is a first-class source — reproducible, unlike the web —
    # but it is a source of a *different kind*, so it gets its own section and
    # its own citation format rather than being folded into the report context.
    forecast_text = state.get("forecast_text", "")

    try:
        answer = ask(
            prompts.SYSTEM_PROMPT,
            prompts.build_answer_prompt(
                question, report_context, web_context, forecast_text, history_context
            ),
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
    # Ход дописывается в историю здесь, в единственном месте, где разговор
    # действительно состоялся: вопрос задан и ответ получен.
    return {"answer": answer, "history": [{"question": question, "answer": answer}]}


def node_out_of_scope(state: AgentState) -> AgentState:
    # Отказ — тоже ход разговора: без него следующее «а почему?» повисает в
    # воздухе, потому что модель не видит, на что отвечала.
    return {
        "answer": prompts.OUT_OF_SCOPE_REPLY,
        "used_reports": False,
        "used_web": False,
        "history": [{"question": state["question"], "answer": prompts.OUT_OF_SCOPE_REPLY}],
    }


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


def _serializer():
    """Сериализатор состояния с явным списком разрешённых своих типов.

    В состоянии едут наши объекты ``Hit`` (результаты поиска по корпусу).
    Восстановление произвольного типа из чекпоинта — это исполнение кода по
    данным из хранилища, поэтому LangGraph требует называть свои типы явно и
    в следующих версиях запретит молчаливое восстановление вовсе. Список
    здесь — ровно наш модуль, и ничего сверх него.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    # ⚠Разрешение даётся по ПОЛНОМУ имени модуля, а не по корню пакета:
    # значение "neftegaz" не покрывает "neftegaz.rag.store", и с ним
    # восстановление молча блокировалось.
    # ⚠Разрешение даётся ПАРОЙ (модуль, имя типа), а не именем модуля: ни
    # "neftegaz", ни "neftegaz.rag.store" сами по себе не покрывают тип, и с
    # ними восстановление молча блокировалось.
    return JsonPlusSerializer(allowed_msgpack_modules=[("neftegaz.rag.store", "Hit")])


def build_checkpointer():
    """Хранилище ходов разговора, выбранное настройкой ``CONVERSATION_MEMORY``.

    Три состояния, и все три — законные развёртывания, а не степени готовности:

    * ``memory`` — в оперативной памяти процесса. Разговор живёт, пока живёт
      процесс; на диск не попадает ничего. Умолчание.
    * ``sqlite`` — файл на диске: разговор переживает перезапуск, но переписка
      заказчика оказывается записанной.
    * ``off`` — памяти нет, каждый вопрос отвечается с чистого листа.

    ★Выбор оставлен тому, кто разворачивает систему, именно потому, что это
    решение не техническое: переписка с аналитиком — данные заказчика, и
    записывать ли их, зависит от его регламента, а не от нашего удобства.

    Неизвестное значение — ошибка, а не тихий откат к умолчанию: опечатка в
    ``CONVERSATION_MEMORY`` иначе выглядела бы как работающая система, у
    которой просто пропала память.
    """
    mode = settings.conversation_memory.strip().lower()
    if mode == "off":
        return None
    if mode == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver(serde=_serializer())
    if mode == "sqlite":
        import sqlite3
        from pathlib import Path

        from langgraph.checkpoint.sqlite import SqliteSaver

        path = Path(settings.checkpoint_db)
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False обязателен: Streamlit обслуживает страницы в
        # разных потоках, и соединение, привязанное к потоку-создателю, падает
        # на втором вопросе.
        connection = sqlite3.connect(str(path), check_same_thread=False)
        saver = SqliteSaver(connection, serde=_serializer())
        saver.setup()
        return saver
    raise ValueError(
        f"unknown CONVERSATION_MEMORY={settings.conversation_memory!r}; "
        "expected one of: memory, sqlite, off"
    )


def new_thread_id() -> str:
    """Идентификатор нового разговора."""
    import uuid

    return uuid.uuid4().hex


def build_graph(checkpointer=None):
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
    return graph.compile(checkpointer=checkpointer)


_COMPILED = None


def answer_question(question: str, thread_id: str | None = None) -> AgentState:
    """Run one question through the agent and return the full final state.

    ``thread_id`` называет разговор. Один и тот же идентификатор продолжает
    беседу: предыдущие ходы приезжают из чекпоинтера и попадают в промпт, так
    что «а на пять лет?» имеет к чему относиться. Без него каждый вопрос
    отвечается с чистого листа — это законный режим для пакетных прогонов и
    демонстраций, где разговора нет и общая память только мешала бы
    воспроизводимости.
    """
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_graph(build_checkpointer())
    initial: AgentState = {"question": question, "used_web": False, "used_reports": False}
    config = {"configurable": {"thread_id": thread_id or new_thread_id()}}
    return _COMPILED.invoke(initial, config=config)
