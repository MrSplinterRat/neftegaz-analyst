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

from neftegaz.agent.graph import answer_question  # noqa: E402
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
def corpus_size() -> int:
    """Number of indexed chunks, or 0 if the index has not been built."""
    try:
        from neftegaz.rag.store import get_store

        return get_store().count()
    except Exception:  # noqa: BLE001 - an absent index is a normal first-run state
        return 0


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Конфигурация")
        st.caption("Всё задаётся через .env — см. .env.example")

        chunks = corpus_size()
        if chunks:
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
                state = answer_question(question)
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
