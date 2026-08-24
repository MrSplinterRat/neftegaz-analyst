"""LLM access, behind one function so the rest of the code never sees a vendor.

Any OpenAI-compatible endpoint works: a hosted model (OpenAI, or a proxy in
front of Claude/GigaChat/YandexGPT) or a local llama.cpp / vLLM server. The
default in `.env.example` is local, so the project starts with no key at all.
"""

from __future__ import annotations

from functools import lru_cache

from neftegaz.config import settings

__all__ = ["get_llm", "ask", "llm_available"]


@lru_cache(maxsize=1)
def get_llm():
    """Build the chat model once per process."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
        max_retries=1,
    )


def ask(system: str, user: str) -> str:
    """One round trip. Returns the reply text."""
    from langchain_core.messages import HumanMessage, SystemMessage

    reply = get_llm().invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = reply.content
    # Some servers return content as a list of parts rather than a string.
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content).strip()


def llm_available() -> bool:
    """Cheap liveness probe, used by the UI and by the demo runner.

    Deliberately does a real (tiny) completion rather than checking a health
    endpoint: a reachable server with no model loaded answers health checks
    fine and fails the only thing we actually need from it.
    """
    try:
        return bool(ask("Отвечай одним словом.", "Скажи: готов"))
    except Exception:  # noqa: BLE001 - any failure means "not available"
        return False
