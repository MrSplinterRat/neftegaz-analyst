"""Streamlit interface for the oil & gas analyst agent.

Deliberately thin: it collects a question, runs the graph, and shows the answer
alongside *which sources were actually consulted*. That last part is not
decoration — requirement 2.4 is about source priority, and a UI that hides
which branch ran makes the property impossible to check by looking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neftegaz.agent.graph import answer_question, new_thread_id  # noqa: E402
from neftegaz.config import settings  # noqa: E402

st.set_page_config(page_title="Нефтегазовый аналитик", page_icon="🛢", layout="wide")

EXAMPLES = [
    "Какой прогноз EIA по добыче нефти в США на следующий год?",
    "Спрогнозируй цену Brent на 3 месяца",
    "Оцени диапазон цен при сокращении добычи ОПЕК+ на 1.5 млн барр./сут",
    "Какие сейчас котировки Brent и что на них влияет?",
    "Посоветуй рецепт борща",
]


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


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Конфигурация")
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
        memory_labels = {
            "memory": "в памяти процесса (не пишется на диск)",
            "sqlite": f"в файле {settings.checkpoint_db}",
            "off": "выключена — каждый вопрос с чистого листа",
        }
        mode = settings.conversation_memory.strip().lower()
        st.markdown(
            f"**Память диалога**\n\n{memory_labels.get(mode, mode)}\n\n"
            f"бюджет истории `{settings.history_budget_chars}` знаков"
        )
        if st.button("Начать разговор заново", use_container_width=True):
            # Новый идентификатор, а не очистка хранилища: прежний разговор
            # остаётся на месте, а этот начинается с чистого листа.
            st.session_state.thread_id = new_thread_id()
            st.session_state.history = []
            st.rerun()

        st.divider()
        st.caption(
            "Приоритет источников: сначала база отраслевых отчётов, "
            "веб-поиск — как дополнение и для актуальных данных."
        )


def render_sources(state: dict) -> None:
    """Show what the agent actually consulted, with scores and links."""
    report_hits = state.get("report_hits") or []
    web_hits = state.get("web_hits") or []

    if not report_hits and not web_hits:
        return

    columns = st.columns(2)
    with columns[0]:
        st.markdown("##### Из базы отчётов")
        if not report_hits:
            st.caption("не использовались")
        for hit in report_hits:
            pages = f"с. {hit.page}" + (f"–{hit.page_end}" if hit.page_end != hit.page else "")
            with st.expander(f"{hit.source_name}, {hit.date}, {pages} · {hit.score:.3f}"):
                st.text(hit.text[:1500])

    with columns[1]:
        st.markdown("##### Из веба")
        if not web_hits:
            st.caption("не использовался")
        for hit in web_hits:
            mark = " ⭐" if hit.preferred else ""
            with st.expander(f"{hit.domain}{mark} — {hit.title[:60]}"):
                st.text(hit.snippet[:1000])
                st.markdown(f"[{hit.url}]({hit.url})")


def main() -> None:
    st.title("🛢 Старший аналитик нефтегазового рынка")
    st.caption(
        "RAG по отраслевым отчётам + веб-поиск + расчётный модуль прогнозирования цен"
    )
    render_sidebar()

    if "history" not in st.session_state:
        st.session_state.history = []
    # Идентификатор разговора живёт в сессии браузера: два открытых окна —
    # два разных разговора, и ходы одного не подмешиваются в другой.
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = new_thread_id()

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
            st.caption("Источники: " + " · ".join(badges))

        render_sources(state)

    st.session_state.history.append({"question": question, "answer": answer})


if __name__ == "__main__":
    main()
