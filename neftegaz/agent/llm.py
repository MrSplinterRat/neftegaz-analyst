"""LLM access, behind one function so the rest of the code never sees a vendor.

Any OpenAI-compatible endpoint works: a hosted model (OpenAI, or a proxy in
front of Claude/GigaChat/YandexGPT) or a local llama.cpp / vLLM server. The
default in `.env.example` is local, so the project starts with no key at all.
"""

from __future__ import annotations

import re
from functools import lru_cache

from neftegaz.config import settings

__all__ = ["get_llm", "ask", "llm_available", "strip_reasoning"]


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


def strip_reasoning(text: str) -> str:
    """Remove a reasoning model's internal monologue from its reply.

    Reasoning models (QwQ, DeepSeek-R1, Qwen3 and friends) emit their working
    inside <think>…</think> before the actual answer. That block must never
    reach a user or a parser:

    * shown to a user it is a multi-kilobyte wall of English deliberation
      sitting on top of the answer;
    * parsed as an answer it is actively misleading, because the model names
      every option it considered while rejecting them. A classifier that
      answers "other" after weighing "forecast" will look like "forecast" to
      any substring search.

    An unterminated <think> (a reply cut off by the token ceiling) drops
    everything from the tag onwards: a truncated monologue contains no answer,
    and keeping it would mean feeding deliberation to whoever asked.
    """
    if "<think>" not in text:
        return text.strip()
    closed = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return closed.split("<think>")[0].strip()


def ask(system: str, user: str) -> str:
    """One round trip. Returns the reply text, without any reasoning block."""
    from langchain_core.messages import HumanMessage, SystemMessage

    reply = get_llm().invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = reply.content
    # Some servers return content as a list of parts rather than a string.
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return strip_reasoning(str(content))


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
