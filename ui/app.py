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
from neftegaz.tools.web import checked_backend  # noqa: E402

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


def panel_settings() -> None:
    """Настройки, разложенные по тому, ЧТО ИМЕННО они портят.

    ★Разложение важнее списка. Температура и число веб-результатов меняют
    следующий ответ и ничего не ломают. Модель эмбеддингов и размер фрагмента
    меняют САМ ИНДЕКС: система продолжит отвечать уверенно, но из индекса,
    собранного по другим правилам. Показать их вперемешку значило бы сделать
    вторые такими же безобидными на вид, как первые.
    """
    st.markdown("### ⚙️ Настройки")

    used = (st.session_state.get("last_state") or {}).get("context_used") or {}
    if used:
        st.markdown("**Чем был заполнен контекст последнего вопроса**")
        # ★Не «бюджет 7000», а сколько из него ушло и на что. Мёртвое число не
        # отвечает на вопрос «почему модель не увидела фрагмент», а это —
        # отвечает, и без чтения кода.
        st.markdown(
            f"- отчёты: **{used.get('reports', 0)}** из {used.get('reports_budget', 0)} знаков "
            f"(фрагментов подано {used.get('fragments_fed', 0)} "
            f"из {used.get('fragments_found', 0)} найденных)\n"
            f"- история разговора: **{used.get('history', 0)}** "
            f"из {used.get('history_budget', 0)}\n"
            f"- веб: **{used.get('web', 0)}** из {used.get('web_budget', 0)}\n"
            f"- расчётный модуль: **{used.get('forecast', 0)}**"
        )
        # ★Числа разметки ступеней стоят рядом с числами контекста намеренно:
        # оба отвечают на вопрос «что на самом деле произошло на этом ходу».
        # Отдельно названа ссылка, не совпавшая ни с одним поданным фрагментом:
        # это не «спорная страница», а «сослались на то, чего не читали».
        marks = (st.session_state.get("last_state") or {}).get("confidence_marks") or {}
        if marks.get("citations"):
            line = (
                f"- ссылок на отчёты в ответе: **{marks['citations']}**, "
                f"из них с пометкой о чтении: **{marks.get('marked', 0)}**"
            )
            if marks.get("unmatched"):
                line += f"\n- ⚠ ссылок мимо поданных фрагментов: **{marks['unmatched']}**"
            st.markdown(line)
    else:
        st.caption("Задайте вопрос — здесь появится, чем именно был заполнен контекст.")

    st.markdown("---")
    st.markdown("**Параметры хода** — действуют со следующего вопроса, ничего не ломают")
    # ★«температура -1.0» — мёртвое число: отрицательного значения у температуры
    # не бывает, это наш признак «не передавать параметр серверу вовсе» (нужен
    # моделям, которые отвергают запрос с температурой целиком). Показывать его
    # как число значит требовать от читателя знания нашего кода.
    temperature = (
        "не передаётся серверу (модель отвергает этот параметр)"
        if settings.llm_temperature < 0
        else f"`{settings.llm_temperature}`"
    )
    st.markdown(
        f"- температура модели {temperature}, "
        f"таймаут `{settings.llm_timeout}` с\n"
        f"- фрагментов из отчётов `RAG_TOP_K = {settings.top_k}`, "
        f"порог близости `RAG_MIN_SCORE = {settings.min_score}`\n"
        f"- веб-результатов `{settings.web_results}`, регион `{settings.web_region}`\n"
        f"- бюджет истории `{settings.history_budget_chars}` знаков, "
        f"ход обрезается до `{settings.history_turn_cap_chars}`"
    )

    st.markdown("**Параметры разговора** — живут при нити, а не глобально")
    st.markdown(
        f"- память диалога `{settings.conversation_memory}`\n"
        f"- подключение разговоров источниками "
        f"`LINK_THREADS = {'включено' if settings.link_threads else 'выключено'}` — "
        "выключено, потому что отрицательный контроль меры не пройден (ОТЧЁТ.md, 3.15б)"
    )

    st.markdown("**★Параметры корпуса** — их правка делает индекс несогласованным")
    st.markdown(
        f"- модель эмбеддингов `{settings.embedding_model}`\n"
        f"- размер фрагмента `{settings.chunk_size}`, перекрытие `{settings.chunk_overlap}`\n"
        f"- коллекция `{settings.collection}`"
    )
    _render_index_stamp()

    st.markdown("---")
    st.markdown("**Ключи и секреты**")
    key = (settings.llm_api_key or "").strip()
    placeholder = key in {"", "not-needed-for-local"}
    st.markdown(
        f"- ключ языковой модели: **{'не задан (локальная модель)' if placeholder else 'задан'}**"
    )
    # ★Значение ключа не показывается никогда и ни в каком виде. Панель
    # настроек — самое естественное место, чтобы секрет утёк в скриншот.
    st.caption("Значения секретов в интерфейсе не показываются ни при каких настройках.")

    st.markdown("---")
    st.markdown("**Исходящий трафик**")
    st.caption(
        "Система предлагается как локальная, поэтому список того, что уходит наружу, "
        "должен быть виден, а не выясняться сниффером. Адреса и ключи здесь не "
        "показываются — только состояние."
    )
    # ★Поисковый сервис называется поимённо, потому что именно туда уезжает
    # текст вопроса пользователя. Строка «веб-поиск включён» без имени адресата
    # выглядит как честная опись и ею не является.
    backend, backend_why = checked_backend()
    web_backend = f"{backend} (через ddgs)" if backend else "адресат не определён"
    web_state = "включено" if backend else f"выключено — {backend_why[:80]}"
    st.markdown(
        f"| куда | зачем | когда | состояние |\n|---|---|---|---|\n"
        f"| языковая модель | сам ответ | каждый вопрос | "
        f"{'локальная' if '127.0.0.1' in settings.llm_base_url else 'внешний endpoint'} |\n"
        f"| {web_backend} | веб-поиск | когда отчётов не хватило | {web_state} |\n"
        f"| Yahoo Finance (yfinance) | история цен Brent | обновление котировок | "
        f"по запуску скрипта |\n"
        f"| HuggingFace (fastembed) | модель эмбеддингов, 241 МБ | первый запуск | "
        f"{'модель уже загружена' if _embedding_model_cached() else 'потребуется загрузка'} |"
    )
    st.caption(
        "Проверка не флагом, а прогоном: `docker run --network none …` — "
        "система поднимается, ищет по отчётам и считает прогноз из кэша цен, "
        "а недоступность модели называет первой строкой ответа."
    )


def _embedding_model_cached() -> bool:
    """Лежит ли модель эмбеддингов на диске — то есть нужна ли загрузка."""
    cache = os.getenv("FASTEMBED_CACHE_PATH", "")
    if not cache:
        return False
    try:
        return any(Path(cache).iterdir())
    except OSError:
        return False


def _render_index_stamp() -> None:
    """Метка рассогласования индекса с настройкой — и она не гаснет сама."""
    from neftegaz.rag.index_stamp import mismatches, read_stamp

    stamp = read_stamp()
    if stamp is None:
        st.warning(
            "Правила сборки индекса неизвестны: отметки рядом с хранилищем нет. "
            "Это не значит «расхождений нет» — значит, сравнить не с чем. "
            "Пересоберите индекс (`python scripts/build_index.py`), чтобы отметка появилась."
        )
        return
    diff = mismatches()
    if not diff:
        st.success(
            f"Индекс собран по текущим правилам "
            f"({stamp.get('chunks', '?')} фрагментов, {str(stamp.get('built_at', ''))[:16]})."
        )
        return
    lines = "\n".join(f"- `{name}`: индекс собран с `{was}`, сейчас `{now}`" for name, was, now in diff)
    st.error(
        "★ИНДЕКС НЕ СОГЛАСОВАН С НАСТРОЙКОЙ. Ответы идут из индекса, собранного "
        "по другим правилам, и выглядят при этом совершенно обычно.\n\n"
        f"{lines}\n\n"
        "Метка не исчезнет сама — она исчезнет пересборкой: "
        "`python scripts/build_index.py --recreate`."
    )


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
    "settings": ("⚙️", "Настройки", panel_settings),
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
        registry.record_turn(
            st.session_state.thread_id,
            question,
            answer,
            # След находок — идентификаторы фрагментов, реально поданных модели
            # на этом ходу. Их считает узел ответа, а не интерфейс: счётчик
            # обязан стоять там, где принимается решение.
            chunk_ids=state.get("fed_chunk_ids") or (),
        )
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
