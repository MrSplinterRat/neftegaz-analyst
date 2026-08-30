"""Configuration, read once from the environment.

Everything tunable lives in `.env` (see `.env.example`). Nothing in this
project reads `os.environ` directly except this module, so there is exactly one
place to look when a deployment behaves unexpectedly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

__all__ = ["Settings", "settings", "ROOT"]


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for one process."""

    # ── LLM ────────────────────────────────────────────────────────────────
    # Any OpenAI-compatible endpoint. The default points at a local llama.cpp
    # server so the project runs with no account and no key; set
    # OPENAI_BASE_URL / OPENAI_API_KEY to use a hosted model instead.
    llm_base_url: str = field(
        default_factory=lambda: _env("OPENAI_BASE_URL", "http://127.0.0.1:8081/v1")
    )
    llm_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", "not-needed-for-local"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "local"))
    # ★Отрицательное значение означает «не передавать параметр серверу вовсе».
    # Нужно для моделей, которые отвергают запрос с температурой целиком
    # (400 invalid_request_error: `temperature` is deprecated for this model).
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.1))
    llm_timeout: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT", 300))

    # ── Embeddings ─────────────────────────────────────────────────────────
    # Local by default: the report corpus must not leave the machine, and a
    # local model removes one more key from the setup path.
    #
    # The default is multilingual and small (~0.5 GB, 384 dims). Multilingual
    # is not optional here: the corpus is in English (EIA/OPEC/IEA publish in
    # English) while questions arrive in Russian, so retrieval is cross-lingual
    # by nature. Set EMBEDDING_MODEL=intfloat/multilingual-e5-large for better
    # quality at ~2.2 GB and roughly 4x the indexing time.
    embedding_model: str = field(
        default_factory=lambda: _env(
            "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    )

    # ── Vector store ───────────────────────────────────────────────────────
    # Embedded Qdrant writing to a local directory: no separate service to run,
    # which is what lets `docker run` be a single command.
    qdrant_path: str = field(
        default_factory=lambda: _env("QDRANT_PATH", str(ROOT / "data" / "qdrant"))
    )
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", ""))
    collection: str = field(default_factory=lambda: _env("QDRANT_COLLECTION", "reports"))

    # ── Retrieval ──────────────────────────────────────────────────────────
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 5))
    # Below this cosine score a hit is treated as "the corpus does not cover
    # this", which is what triggers the web fallback in the routing logic.
    #
    # 0.55 is measured, not guessed. On the EIA STEO corpus with the default
    # embedding model, best-hit scores separate cleanly:
    #   industry questions   0.700 … 0.819   (Russian query, English corpus)
    #   off-topic questions  0.140 … 0.452
    # The gap is 0.248 wide; the threshold sits in the middle of it, leaning
    # towards recall — a weak fragment shown to the model costs little, while a
    # missed one sends the agent to the web for something the corpus contained.
    # ★Re-measure after changing the embedding model: cosine scores are not
    # comparable across models, and a threshold carried over blindly will
    # either flood the context or silently empty it.
    min_score: float = field(default_factory=lambda: _env_float("RAG_MIN_SCORE", 0.55))
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 1200))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 200))

    # ── Web search ─────────────────────────────────────────────────────────
    web_results: int = field(default_factory=lambda: _env_int("WEB_RESULTS", 5))
    web_region: str = field(default_factory=lambda: _env("WEB_REGION", "ru-ru"))

    # ── Откуда берётся эластичность ────────────────────────────────────────
    # measured    — оценить на корпусе отчётов (умолчание): квартальные ряды
    #               добычи, потребления, запасов и цены из таблиц STEO;
    # literature  — взять числа ниже, ничего не считая.
    # ★Измерение с откатом, а не вместо: если корпус пуст или оценка вышла
    # непригодной (положительная, либо интервал накрывает ноль), расчёт берёт
    # литературное число и ГОВОРИТ об этом в ответе. Молчаливый откат означал бы,
    # что пользователь не знает, на чём стоит увиденная им цифра.
    elasticity_source: str = field(default_factory=lambda: _env("ELASTICITY_SOURCE", "measured"))

    # ── Сценарий предложения: эластичность спроса по цене ──────────────────
    # Весь сценарный расчёт («что будет с ценой, если добыча упадёт на N млн
    # барр./сут») стоит на этих числах, поэтому они здесь, а не в модуле:
    # рецензент должен иметь возможность не согласиться, не трогая код.
    #
    # ★★В сценарной формуле стоит НЕ эластичность спроса, хотя выглядит она так.
    # Формуле нужна эластичность РЫНОЧНОГО КЛИРИНГА: на сколько процентов должна
    # сдвинуться цена, чтобы рынок поглотил сдвиг ПРЕДЛОЖЕНИЯ на процент. Между
    # ней и эластичностью спроса стоит БУФЕР ЗАПАСОВ: разрыв закрывается не
    # только сокращением потребления, но и расходом хранилищ, а хранилища
    # работают как дополнительное предложение и гасят ценовой отклик.
    #
    # Измерение на корпусе STEO: эластичность спроса −0.11, эластичность
    # клиринга −0.31 — втрое по модулю. Подставив первую вместо второй, расчёт
    # завышал отклик на шок −2 млн барр./сут с ×1.066 до ×1.219.
    #
    # −0.31 это ИЗМЕРЕННОЕ квартальное значение (регрессия отклика цены на сдвиг
    # добычи, 9 разностей, 95% ДИ −0.71…−0.20).
    demand_elasticity_short: float = field(
        default_factory=lambda: _env_float("DEMAND_ELASTICITY_SHORT", -0.31)
    )
    # ⚠Длинный конец НЕ измерен: корпус накрывает полтора года, длинного
    # горизонта в нём нет. Взято литературное −0.30 для эластичности СПРОСА, и
    # вот почему это осмысленно именно для длинного горизонта: за годы буфер
    # запасов не работает (хранилища конечны и успевают пополниться), поэтому
    # клиринг сходится к спросу.
    # ★Численно короткий и длинный конец почти совпали — значит на этих данных
    # мы горизонты НЕ РАЗЛИЧАЕМ, и интерполяция между ними почти плоская. Это
    # честное отражение незнания, а не свойство рынка.
    demand_elasticity_long: float = field(
        default_factory=lambda: _env_float("DEMAND_ELASTICITY_LONG", -0.30)
    )
    # Горизонты, между которыми эластичность интерполируется линейно; за
    # пределами — насыщение. 90 дней это квартал, 1825 — пять лет.
    elasticity_short_days: int = field(
        default_factory=lambda: _env_int("ELASTICITY_SHORT_DAYS", 90)
    )
    elasticity_long_days: int = field(
        default_factory=lambda: _env_int("ELASTICITY_LONG_DAYS", 1825)
    )
    # Полоса неопределённости самой эластичности: коридор прогноза расширяется
    # ровно на это незнание — сценарий добавляет гипотезу, а гипотеза не может
    # сузить доверительный интервал.
    # ★При ELASTICITY_SOURCE=measured полоса берётся из доверительного интервала
    # оценки (край со стороны более сильного отклика), и это значение не
    # используется. Оно остаётся умолчанием для литературного режима: −0.20 это
    # ближний край измеренного интервала.
    demand_elasticity_band: float = field(
        default_factory=lambda: _env_float("DEMAND_ELASTICITY_BAND", -0.20)
    )
    # Мировое предложение жидких углеводородов, млн барр./сут. Значение по
    # умолчанию — порядок, а не факт дня; при подключённом ряду поставщика
    # его следует брать из данных.
    global_supply_mb_d: float = field(
        default_factory=lambda: _env_float("GLOBAL_SUPPLY_MB_D", 102.0)
    )

    # ── Память диалога ─────────────────────────────────────────────────────
    # Где хранятся ходы разговора между вопросами:
    #   memory — в оперативной памяти процесса (умолчание): разговор живёт,
    #            пока живёт процесс, на диск не попадает ничего;
    #   sqlite — в файле, разговор переживает перезапуск;
    #   off    — памяти нет, каждый вопрос отвечается с чистого листа.
    #
    # ★Умолчание НЕ на диске намеренно. Переписка с аналитиком — это данные
    # заказчика, и решение о том, писать ли их на диск, принимает тот, кто
    # разворачивает систему, а не тот, кто её писал.
    conversation_memory: str = field(default_factory=lambda: _env("CONVERSATION_MEMORY", "memory"))
    checkpoint_db: str = field(
        default_factory=lambda: _env("CHECKPOINT_DB", str(ROOT / "data" / "conversations.sqlite"))
    )
    # Бюджет знаков на всю историю, попадающую в промпт. ★Жёсткое число, а не
    # «сколько влезет»: контекст уже ронял систему один раз, и лимит обязан
    # быть виден в конфигурации, а не выведен из размера окна модели.
    # Старые ходы отбрасываются целиком, без пересказа: пересказ означал бы
    # ещё один вызов модели и недетерминизм там, где его быть не должно.
    history_budget_chars: int = field(
        default_factory=lambda: _env_int("HISTORY_BUDGET_CHARS", 4000)
    )
    # Потолок на ОДИН ход. Без него единственный длинный ответ съедает весь
    # бюджет и вытесняет всю остальную историю.
    history_turn_cap_chars: int = field(
        default_factory=lambda: _env_int("HISTORY_TURN_CAP_CHARS", 1200)
    )

    # ── Data locations ─────────────────────────────────────────────────────
    reports_dir: str = field(
        default_factory=lambda: _env("REPORTS_DIR", str(ROOT / "data" / "reports"))
    )
    prices_csv: str = field(
        default_factory=lambda: _env("PRICES_CSV", str(ROOT / "data" / "prices" / "brent.csv"))
    )


settings = Settings()
