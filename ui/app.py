"""Streamlit interface for the oil & gas analyst agent.

Deliberately thin: it collects a question, runs the graph, and shows the answer
alongside *which sources were actually consulted*. That last part is not
decoration — requirement 2.4 is about source priority, and a UI that hides
which branch ran makes the property impossible to check by looking.

★РАСКЛАДКА: РЕЛЬС ИКОНОК СЛЕВА, ПАНЕЛЬ ВЫЕЗЖАЕТ ПО КНОПКЕ (образец — Vivaldi).
Прежде вся справочная информация висела в постоянно открытой боковой колонке.
Она не нужна во время разговора: читается один раз, а место занимает всегда.
Теперь слева стоит узкая полоса иконок, и каждая раскрывает свою панель;
повторный клик по активной иконке — сворачивает. Разговор при этом получает всю
оставшуюся ширину, а справка доступна за один клик.

Штатный боковой блок Streamlit (``st.sidebar``) здесь не используется вовсе:
он один на приложение и не делится на «полосу иконок» и «содержимое панели».
Раскладка собрана из обычных колонок, поэтому ведёт себя предсказуемо и не
зависит от внутренней вёрстки Streamlit — кроме одного правила CSS, которое
прижимает рельс к верху при прокрутке.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neftegaz.agent.graph import answer_question, new_thread_id  # noqa: E402
from neftegaz.agent.threads import (  # noqa: E402
    MIN_QUERY_CHARS,
    fts_query,
    get_registry,
    registry_unavailable_reason,
)
from neftegaz.config import settings  # noqa: E402

st.set_page_config(
    page_title="Нефтегазовый аналитик",
    page_icon="🛢",
    layout="wide",
    # Своя раскладка занимает место штатного бокового блока, поэтому он свёрнут
    # и не используется: две панели слева спорили бы за одно и то же место.
    initial_sidebar_state="collapsed",
)

EXAMPLES = [
    "Какой прогноз EIA по добыче нефти в США на следующий год?",
    "Спрогнозируй цену Brent на 3 месяца",
    "Оцени диапазон цен при сокращении добычи ОПЕК+ на 1.5 млн барр./сут",
    "Какие сейчас котировки Brent и что на них влияет?",
    "Посоветуй рецепт борща",
]

# ★ВЕСЬ CSS — ОДНОЙ КОНСТАНТОЙ, А НЕ РОССЫПЬЮ ПО КОДУ. Правила ниже трогают
# внутреннюю вёрстку Streamlit, то есть могут перестать действовать при его
# обновлении. Собранные в одном месте, они переживают обновление как одна
# заметная поломка вида, а не как пять разных мелких.
RAIL_CSS = """
<style>
/* Штатная кнопка бокового блока увела бы пользователя ко второй панели,
   которой мы не пользуемся. */
[data-testid="stSidebarCollapsedControl"] { display: none; }

/* ★ЯКОРЬ — ИМЕНОВАННЫЙ КОНТЕЙНЕР (st.container(key=…) даёт класс st-key-…),
   а не порядковый номер колонки. Порядковый селектор молча промахивается,
   когда Streamlit меняет вложенность или когда панель закрыта и колонок две,
   а не три: правило продолжает существовать и ничего не красить. */
.st-key-rail {
    position: sticky;
    top: 3.2rem;
}

/* Иконки рельса: квадратные, без рамки, крупный знак. Кнопка Streamlit по
   умолчанию тянется под текст и выглядит как кнопка формы — здесь нужен
   переключатель. Отступ между ними убран: в рельсе они читаются как один
   столбец, а не как список отдельных команд. */
.st-key-rail div[data-testid="stVerticalBlock"] { gap: 0.25rem; }
.st-key-rail button {
    width: 2.7rem;
    min-width: 2.7rem;
    height: 2.7rem;
    padding: 0;
    font-size: 1.2rem;
    line-height: 1;
    border: none;
    border-radius: 0.5rem;
    background: transparent;
}
.st-key-rail button:hover {
    background: rgba(128, 128, 128, 0.16);
}
/* Активная панель. Заливка Streamlit для primary слишком криклива для
   переключателя, который виден постоянно, поэтому переопределяется целиком. */
.st-key-rail button[kind="primary"],
.st-key-rail button[kind="primary"]:hover,
.st-key-rail button[kind="primary"]:focus {
    background: rgba(128, 128, 128, 0.22) !important;
    box-shadow: inset 2px 0 0 0 rgb(255, 75, 75) !important;
    color: inherit !important;
}

/* Панель отделена от разговора линией, а не пустотой: пустота на широком
   экране читается как случайный отступ. */
.st-key-panel {
    border-right: 1px solid rgba(128, 128, 128, 0.25);
    padding-right: 1.1rem;
    padding-left: 0.5rem;
}
.st-key-panel h3 { margin-top: 0; }
</style>
"""


@st.cache_resource(show_spinner=False)
def corpus_size() -> tuple[int, str | None]:
    """Number of indexed chunks, plus the reason the count is unavailable.

    Two very different states used to collapse into the same zero: an index
    that has not been built yet, and an index that exists but cannot be
    opened. The second one is routine here — embedded Qdrant admits a single
    writer, so a concurrent scripts/run_demo.py, a second container or another
    UI instance holds the storage folder. Reporting that as "the corpus is
    empty, go build it" sends the user to rebuild an index that is already
    there, so the cause is carried out instead of being swallowed.
    """
    try:
        from neftegaz.rag.store import get_store

        return get_store().count(), None
    except Exception as exc:  # noqa: BLE001 - the cause is reported, not hidden
        return 0, str(exc)


# ── содержимое панелей ─────────────────────────────────────────────────────


def panel_info() -> None:
    """Конфигурация: то, что раньше висело в боковом блоке постоянно."""
    st.markdown("### ℹ️ Конфигурация")
    st.caption("Всё задаётся через .env — см. .env.example")

    chunks, unavailable = corpus_size()
    if unavailable:
        # Отказ не кэшируем: держатель блокировки уходит, и следующий
        # прогон страницы должен увидеть базу, а не замороженную ошибку.
        corpus_size.clear()
        st.warning(
            "База отчётов сейчас недоступна — индекс не открывается.\n\n"
            "Обычная причина: встроенный Qdrant допускает одного "
            "писателя, а каталог занят другим процессом "
            "(scripts/run_demo.py, второй контейнер, ещё один UI).\n\n"
            f"Ответ хранилища: {unavailable}"
        )
    elif chunks:
        st.success(f"База отчётов: {chunks} фрагментов")
    else:
        st.warning(
            "База отчётов пуста.\n\n"
            "Соберите её:\n"
            "1. `python scripts/fetch_reports.py`\n"
            "2. `python scripts/build_index.py`"
        )

    st.markdown(
        f"""
**LLM**
`{settings.llm_model}`
`{settings.llm_base_url}`

**Эмбеддинги**
`{settings.embedding_model}`

**Поиск**
top-k `{settings.top_k}`, порог близости `{settings.min_score}`
        """
    )

    st.divider()
    st.caption(
        "Приоритет источников: сначала база отраслевых отчётов, "
        "веб-поиск — как дополнение и для актуальных данных."
    )


def panel_sources() -> None:
    """Источники последнего ответа — с оценками близости и текстом фрагмента."""
    st.markdown("### 📚 Источники ответа")
    state = st.session_state.get("last_state") or {}
    report_hits = state.get("report_hits") or []
    web_hits = state.get("web_hits") or []

    if not report_hits and not web_hits:
        st.caption(
            "Пока пусто: панель наполняется после ответа, "
            "в котором агент обращался к отчётам или вебу."
        )
        return

    st.markdown("**Из базы отчётов**")
    if not report_hits:
        st.caption("не использовались")
    for hit in report_hits:
        pages = f"с. {hit.page}" + (f"–{hit.page_end}" if hit.page_end != hit.page else "")
        with st.expander(f"{hit.source_name}, {hit.date}, {pages} · {hit.score:.3f}"):
            st.text(hit.text[:1500])

    st.markdown("**Из веба**")
    if not web_hits:
        st.caption("не использовался")
    for hit in web_hits:
        mark = " ⭐" if hit.preferred else ""
        with st.expander(f"{hit.domain}{mark} — {hit.title[:60]}"):
            st.text(hit.snippet[:1000])
            st.markdown(f"[{hit.url}]({hit.url})")


def start_new_thread() -> None:
    """Новый идентификатор, а не очистка хранилища.

    Прежний разговор остаётся на месте и теперь ДОСТИЖИМ: он есть в реестре, и
    к нему можно вернуться. До реестра «начать заново» означало «потерять».
    """
    st.session_state.thread_id = new_thread_id()
    st.session_state.history = []
    st.session_state.last_state = {}


def open_thread(thread_id: str) -> None:
    """Переключиться на разговор и поднять его ходы из базы.

    ★Ходы читаются из реестра, а не из состояния браузера: в другой вкладке их
    не было бы вовсе, а после перезапуска не было бы нигде.
    """
    registry = get_registry()
    st.session_state.thread_id = thread_id
    st.session_state.history = (
        [{"question": t["question"], "answer": t["answer"]} for t in registry.turns(thread_id)]
        if registry
        else []
    )
    st.session_state.last_state = {}


def panel_conversation() -> None:
    """Разговоры: список, переключение, переименование, удаление."""
    st.markdown("### 💬 Разговоры")

    memory_labels = {
        "memory": "в памяти процесса — перезапуск стирает",
        "sqlite": f"в файле `{settings.checkpoint_db}` — переживает перезапуск",
        "off": "выключена — каждый вопрос с чистого листа",
    }
    mode = settings.conversation_memory.strip().lower()
    st.caption(
        f"память диалога: {memory_labels.get(mode, mode)}; "
        f"бюджет истории {settings.history_budget_chars} знаков, "
        f"ход обрезается до {settings.history_turn_cap_chars}"
    )

    if st.button("➕ Новый разговор", use_container_width=True):
        start_new_thread()
        st.rerun()

    reason = registry_unavailable_reason()
    if reason:
        # Выключенный список обязан объяснять себя: молчащая панель
        # неотличима от сломанной.
        st.info(reason)
        st.markdown(f"**Текущий разговор**\n\n`{st.session_state.get('thread_id', '—')}`")
        st.caption(f"ходов в этом окне: {len(st.session_state.get('history', []))}")
        return

    registry = get_registry()
    current = st.session_state.get("thread_id", "")
    threads = registry.list_threads()

    if not threads:
        st.caption("Разговоров пока нет — задайте первый вопрос.")
        return

    st.markdown("---")
    for info in threads:
        active = info.thread_id == current
        row, trash = st.columns([5, 1], gap="small")
        label = ("● " if active else "") + info.title
        if row.button(
            label,
            key=f"thread_open_{info.thread_id}",
            use_container_width=True,
            type="primary" if active else "secondary",
            help=f"{info.turns} ход(ов), последний {info.updated_at[:16].replace('T', ' ')}",
        ):
            open_thread(info.thread_id)
            st.rerun()
        if trash.button("🗑", key=f"thread_del_{info.thread_id}", help="Удалить разговор"):
            # Подтверждение отдельным ходом: удаление настоящее и необратимое,
            # а промах по кнопке в узкой панели стои́т одного клика.
            st.session_state.pending_delete = info.thread_id
            st.rerun()

    pending = st.session_state.get("pending_delete", "")
    if pending:
        doomed = registry.get(pending)
        st.warning(f"Удалить «{doomed.title if doomed else pending}» вместе со всеми ходами?")
        yes, no = st.columns(2)
        if yes.button("Удалить", key="thread_del_yes", use_container_width=True):
            registry.delete(pending)
            st.session_state.pending_delete = ""
            if pending == current:
                start_new_thread()
            st.rerun()
        if no.button("Отмена", key="thread_del_no", use_container_width=True):
            st.session_state.pending_delete = ""
            st.rerun()

    st.markdown("---")
    st.markdown("**Переименовать текущий**")
    known = registry.get(current)
    new_title = st.text_input(
        "Название разговора",
        value=known.title if known else "",
        key="thread_rename_field",
        label_visibility="collapsed",
        placeholder="разговор начнётся с первого вопроса",
        disabled=known is None,
    )
    if st.button("Переименовать", use_container_width=True, disabled=known is None):
        if registry.rename(current, new_title):
            st.rerun()
        else:
            st.warning("Название не может быть пустым.")


def run_search(query: str) -> None:
    """Выполнить поиск и запомнить запрос вместе с числом найденного."""
    registry = get_registry()
    hits = registry.search_turns(query)
    registry.record_search(query, len(hits))
    st.session_state.search_hits = hits
    st.session_state.search_last = query


def panel_search() -> None:
    """Сквозной поиск по разговорам и история запросов."""
    st.markdown("### 🔎 Поиск по разговорам")

    reason = registry_unavailable_reason()
    if reason:
        st.info(reason)
        return

    registry = get_registry()
    st.caption(
        "Ищем по вопросам и ответам ВСЕХ разговоров — не по отчётам: у них разный "
        f"провенанс и разная цена ошибки. Запрос — от {MIN_QUERY_CHARS} букв; поиск "
        "подстрочный, поэтому находит и другую словоформу."
    )

    with st.form("search_form"):
        query = st.text_input(
            "Запрос",
            key="search_field",
            label_visibility="collapsed",
            placeholder="слово или несколько",
        )
        go = st.form_submit_button("Искать", use_container_width=True)
    if go:
        if fts_query(query):
            run_search(query)
            st.rerun()
        else:
            # ★Не «ничего не найдено»: короткий запрос не искался вовсе, и
            # выдать пустоту значило бы соврать про содержимое разговоров.
            st.warning(
                f"Слишком короткий запрос: нужно хотя бы {MIN_QUERY_CHARS} буквы подряд. "
                "Поиск идёт по тройкам символов, и сопоставлять меньшее не с чем."
            )

    last = st.session_state.get("search_last", "")
    hits = st.session_state.get("search_hits", [])
    if last:
        st.markdown(f"**Найдено: {len(hits)}** по запросу «{last}»")
        for hit in hits:
            with st.container(border=True):
                st.caption(
                    f"{hit.thread_title} · ход {hit.ordinal} · "
                    f"{hit.asked_at[:16].replace('T', ' ')}"
                )
                st.markdown(f"**{hit.question}**")
                st.markdown(hit.answer)
                if st.button(
                    "Перейти в разговор",
                    key=f"search_go_{hit.thread_id}_{hit.ordinal}",
                    use_container_width=True,
                ):
                    open_thread(hit.thread_id)
                    st.rerun()

    history = registry.list_searches()
    if not history:
        return

    st.markdown("---")
    head, clear = st.columns([3, 2], gap="small")
    head.markdown("**История поиска**")
    if clear.button("Очистить всё", key="history_clear", use_container_width=True):
        registry.clear_searches()
        st.rerun()

    for record in history:
        again, drop = st.columns([5, 1], gap="small")
        if again.button(
            record.query,
            key=f"history_run_{record.query}",
            use_container_width=True,
            help=f"найдено {record.hits}, последний раз {record.last_run[:16].replace('T', ' ')}",
        ):
            run_search(record.query)
            st.rerun()
        if drop.button("🗑", key=f"history_del_{record.query}", help="Забыть запрос"):
            # Удаление настоящее: строка уходит из базы, а не помечается
            # скрытой. Это история человека, и «удалено» значит удалено.
            registry.forget_search(record.query)
            st.rerun()


def panel_corpus() -> None:
    """Корпус отчётов: какие файлы лежат в основе ответов."""
    st.markdown("### 📄 Корпус отчётов")
    directory = settings.reports_dir
    try:
        names = sorted(n for n in os.listdir(directory) if n.lower().endswith(".pdf"))
    except OSError as exc:
        st.warning(f"Каталог отчётов недоступен: {exc}")
        return

    if not names:
        st.warning(
            f"В `{directory}` нет ни одного PDF.\n\n"
            "Соберите корпус: `python scripts/fetch_reports.py`"
        )
        return

    st.caption(f"{directory} — {len(names)} файлов")
    for name in names:
        size = os.path.getsize(os.path.join(directory, name)) / 1e6
        st.markdown(f"- `{name}` · {size:.1f} МБ")

    st.divider()
    st.caption(
        "Как эти файлы читаются и что означают ступени достоверности — "
        "в ОТЧЁТ.md, разделы 3.7–3.9."
    )


def panel_forecast() -> None:
    """Расчётный модуль: допущения, на которых стоит прогноз."""
    st.markdown("### 📈 Расчётный модуль")
    st.markdown(
        f"""
**Мировое предложение**
`{settings.global_supply_mb_d}` млн барр./сут — база для доли шока

**Эластичность**
источник `{settings.elasticity_source}`
короткий конец `{settings.demand_elasticity_short}`,
длинный `{settings.demand_elasticity_long}`

**История цен**
`{settings.prices_csv}`
        """
    )
    st.caption(
        "При источнике measured короткий конец не берётся из настройки, "
        "а измеряется на корпусе отчётов; настройка остаётся запасным значением."
    )


# Порядок здесь — порядок иконок в рельсе. Ключ живёт в состоянии сессии,
# поэтому переименование ключа сбросит открытую панель, а не сломает её.
PANELS: dict[str, tuple[str, str, object]] = {
    "info": ("ℹ️", "Конфигурация", panel_info),
    "sources": ("📚", "Источники ответа", panel_sources),
    "conversation": ("💬", "Разговоры", panel_conversation),
    "search": ("🔎", "Поиск по разговорам", panel_search),
    "corpus": ("📄", "Корпус отчётов", panel_corpus),
    "forecast": ("📈", "Расчётный модуль", panel_forecast),
}

DEFAULT_PANEL = "info"


def render_rail() -> None:
    """Полоса иконок: тумблер панели сверху, переключатели под ним.

    ★Клик меняет состояние и перезапускает прогон страницы. Иначе ширины
    колонок остались бы прежними: они задаются ДО того, как станет известно о
    нажатии, и панель открылась бы внутри полосы шириной в одну иконку.
    """
    open_panel = st.session_state.get("panel")

    # Тумблер: закрывает открытую панель или возвращает последнюю открытую.
    if st.button("☰", key="rail_toggle", help="Показать или скрыть панель"):
        if open_panel:
            st.session_state.last_panel = open_panel
            st.session_state.panel = None
        else:
            st.session_state.panel = st.session_state.get("last_panel", DEFAULT_PANEL)
        st.rerun()

    st.write("")  # отбивка тумблера от переключателей

    for key, (icon, title, _) in PANELS.items():
        active = key == open_panel
        if st.button(
            icon,
            key=f"rail_{key}",
            help=title,
            type="primary" if active else "secondary",
        ):
            # Повторный клик по активной иконке сворачивает панель — так же,
            # как в браузере, откуда взята раскладка.
            st.session_state.panel = None if active else key
            st.session_state.last_panel = key
            st.rerun()


def render_chat() -> None:
    """Разговор: ходы, поле ввода, ответ агента."""
    st.title("🛢 Старший аналитик нефтегазового рынка")
    st.caption("RAG по отраслевым отчётам + веб-поиск + расчётный модуль прогнозирования цен")

    with st.expander("Примеры запросов"):
        for example in EXAMPLES:
            st.markdown(f"- {example}")

    question = st.chat_input("Вопрос по нефтегазовому рынку…")

    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])

    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Ищу в отчётах, при необходимости — в вебе…"):
            try:
                state = answer_question(question, thread_id=st.session_state.thread_id)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Ошибка: {exc}")
                return

        answer = state.get("answer", "")
        st.markdown(answer)

        route = state.get("route", "")
        badges = []
        if state.get("used_reports"):
            badges.append("отчёты")
        if state.get("used_web"):
            badges.append("веб")
        if route == "forecast":
            badges.append("расчётный модуль")
        if badges:
            # ★Ссылка на панель, а не пересказ источников прямо здесь: список
            # фрагментов под каждым ответом растил страницу быстрее самого
            # разговора и оттеснял его вверх.
            st.caption("Источники: " + " · ".join(badges) + " — подробности в панели 📚")

    st.session_state.history.append({"question": question, "answer": answer})
    # ★Ход записывает ИНТЕРФЕЙС, а не answer_question. Реестр — это разговоры
    # пользователя, а не журнал вызовов функции: запись внутри агента заводила
    # бы «разговор» на каждый тест и каждый пакетный прогон, и список открылся
    # бы сотней безымянных строк (замер 31.08: в чекпойнтере их накопилось 96).
    registry = get_registry()
    if registry is not None:
        registry.record_turn(st.session_state.thread_id, question, answer)
    # Последнее состояние живёт отдельно от списка ходов: панель источников
    # показывает, чем отвечен ПОСЛЕДНИЙ вопрос, и хранить ради этого найденные
    # фрагменты всех прошлых ходов незачем.
    st.session_state.last_state = state


def main() -> None:
    st.html(RAIL_CSS)

    if "history" not in st.session_state:
        st.session_state.history = []
    # Идентификатор разговора живёт в сессии браузера: два открытых окна —
    # два разных разговора, и ходы одного не подмешиваются в другой.
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = new_thread_id()
    if "panel" not in st.session_state:
        st.session_state.panel = None

    open_panel = st.session_state.panel
    if open_panel:
        rail, panel, chat = st.columns([0.5, 4.0, 9.5], gap="small")
    else:
        rail, chat = st.columns([0.5, 13.5], gap="small")
        panel = None

    with rail, st.container(key="rail"):
        render_rail()

    if panel is not None:
        with panel, st.container(key="panel"), st.container(height=720, border=False):
            PANELS[open_panel][2]()

    with chat:
        render_chat()


if __name__ == "__main__":
    main()
