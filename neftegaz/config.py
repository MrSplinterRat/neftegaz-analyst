"""Configuration, read once from the environment.

Everything tunable lives in `.env` (see `.env.example`). Nothing in this
project reads `os.environ` directly except this module, so there is exactly one
place to look when a deployment behaves unexpectedly.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

__all__ = [
    "Settings",
    "settings",
    "env_settings",
    "ROOT",
    "TURN_PARAMETERS",
    "TurnParameter",
    "turn_parameter",
    "read_overrides",
    "set_turn_parameter",
    "reset_turn_parameter",
    "overrides_path",
]


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


def _env_bool(name: str, default: bool) -> bool:
    """Булева настройка. Неизвестное значение — ошибка, а не тихое «нет».

    Тихий откат превратил бы опечатку в выключенную возможность, о которой
    никто не узнает: система выглядела бы исправной и просто ничего не делала.
    """
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean (true/false), got {raw!r}")


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

    # ── Бюджет контекста ───────────────────────────────────────────────────
    # Сколько знаков собирающая сторона готова отдать каждому виду источника.
    # ★Потолок держит СОБИРАЮЩИЙ, а не модель: сколько окна ни дай, число
    # найденного умножается на длину найденного, и обе величины растут от
    # настроек поиска. Однажды это уже кончилось ответом-свалкой на
    # 500 context_length_exceeded.
    #
    # Числа стояли константами в `neftegaz/agent/graph.py` — то есть снаружи
    # системы их не было видно вовсе, хотя именно они отвечают на вопрос
    # «почему модель не увидела найденный фрагмент».
    report_budget_chars: int = field(
        default_factory=lambda: _env_int("REPORT_BUDGET_CHARS", 7000)
    )
    web_budget_chars: int = field(default_factory=lambda: _env_int("WEB_BUDGET_CHARS", 4000))
    # Потолок на ОДИН фрагмент: без него один длинный фрагмент съедает бюджет
    # целиком и вытесняет остальные.
    fragment_cap_chars: int = field(default_factory=lambda: _env_int("FRAGMENT_CAP_CHARS", 1800))

    # ── Web search ─────────────────────────────────────────────────────────
    web_results: int = field(default_factory=lambda: _env_int("WEB_RESULTS", 5))
    web_region: str = field(default_factory=lambda: _env("WEB_REGION", "ru-ru"))
    # ★ОДИН сервис, названный поимённо. Библиотека `ddgs` по умолчанию работает
    # в режиме `auto`, и это НЕ «DuckDuckGo с запасными вариантами»: она сама
    # выбирает, к какому из семи сервисов обратиться, и текст вопроса
    # пользователя уезжает туда, куда мы не выбирали и не можем назвать заранее.
    # Для системы, которая сдаётся как локальная, это недопустимо: адресат
    # обязан быть один и обязан быть назван в документации.
    # ⚠Значение сверяется со списком движков самой библиотеки ДО запроса —
    # неизвестное имя `ddgs` молча превращает обратно в `auto`
    # (см. neftegaz/tools/web.py, _checked_backend).
    web_backend: str = field(default_factory=lambda: _env("WEB_BACKEND", "brave"))

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
    #   sqlite — в файле (умолчание): разговор переживает перезапуск, работают
    #            список разговоров и сквозной поиск по ним;
    #   memory — в оперативной памяти процесса: на диск не попадает ничего, но
    #            ни списка разговоров, ни поиска по ним нет;
    #   off    — памяти нет, каждый вопрос отвечается с чистого листа.
    #
    # ★НАСТРОЙКА РЕШАЕТ ОДИН ВОПРОС: попадает ли переписка с аналитиком на диск.
    # Переписка — данные заказчика, и решение принимает тот, кто разворачивает
    # систему, а не тот, кто писал код. Реестр разговоров и история поиска
    # (`neftegaz.agent.threads`) включены ровно при `sqlite` и при остальных
    # значениях говорят в интерфейсе, что их нет и почему, — обойти настройку
    # молча они не могут по устройству.
    #
    # ⚠УМОЛЧАНИЕ ИЗМЕНЕНО С `memory` НА `sqlite` 31.08.2026, и это решение, а не
    # недосмотр. Множественность разговоров и поиск по ним вошли в поставку, а с
    # `memory` половина кнопок интерфейса выключена — то есть прежнее умолчание
    # отдавало бы урезанный продукт. Кому нельзя писать на диск, ставит `memory`
    # одной правкой, и система об этом честно говорит.
    conversation_memory: str = field(default_factory=lambda: _env("CONVERSATION_MEMORY", "sqlite"))
    # ★ПОДКЛЮЧЕНИЕ РАЗГОВОРОВ ИСТОЧНИКАМИ ССЫЛОК — ВЫКЛЮЧЕНО, И ЭТО ИЗМЕРЕНО.
    #
    # Возможность написана и покрыта тестами: из подключённого разговора едут
    # только идентификаторы найденных ранее фрагментов, они соревнуются с
    # обычными кандидатами по той же мере и без всякой надбавки.
    #
    # Польза измерена и есть: на двух переформулировках из трёх нужный фрагмент
    # входит в top-5, куда без подключения не попадал. Но ОТРИЦАТЕЛЬНЫЙ КОНТРОЛЬ
    # НЕ ПРОЙДЕН: на вопросах, к подключённому разговору отношения не имеющих,
    # выдача менялась в 1 случае из 6 (17%), а требовалось «около нуля».
    #
    # Незамеренная возможность в поставке хуже отсутствующей: она выглядит
    # работающей. Поэтому умолчание — «выключено», а не «почти работает».
    # Включается `LINK_THREADS=true` тем, кто согласен с этой ценой.
    link_threads: bool = field(default_factory=lambda: _env_bool("LINK_THREADS", False))
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
# Снимок того, что говорят `.env` и умолчания, — БЕЗ правок из интерфейса.
# ★Нужен затем, чтобы правка из интерфейса не превращалась в тихое расхождение:
# человек, который открыл `.env` и прочёл `RAG_TOP_K=5`, обязан узнать в панели,
# что в работе сейчас другое число, а не гадать, почему система ведёт себя иначе,
# чем написано в её конфигурации.
env_settings = Settings()


# ── параметры хода, правимые из интерфейса ─────────────────────────────────
#
# ★СПИСОК ЗАКРЫТЫЙ, И В ЭТОМ ВЕСЬ СМЫСЛ. Настройки системы делятся по тому, что
# они ломают: параметры ХОДА действуют со следующего вопроса и не трогают
# ничего, кроме него; параметры КОРПУСА (модель эмбеддингов, размер фрагмента,
# коллекция) делают уже собранный индекс несогласованным с настройкой. Первые
# можно править из интерфейса безопасно, вторые — нельзя, и разделение обязано
# быть не советом в документации, а списком в коде: то, чего здесь нет, из
# интерфейса не правится вовсе.


@dataclass(frozen=True)
class TurnParameter:
    """Одна правимая настройка: имя поля, имя в `.env` и границы допустимого."""

    field: str
    env: str
    label: str
    kind: str  # "int" | "float" | "text"
    low: float | None = None
    high: float | None = None
    pattern: str | None = None
    note: str = ""

    def parse(self, raw):
        """Разобрать введённое значение или отказать с внятной причиной.

        ★Отказ громкий, а не тихий откат к прежнему. Опечатка, молча
        превращённая в умолчание, выглядит как исправная система, которая
        почему-то делает не то.
        """
        if self.kind == "text":
            value = str(raw).strip()
            if self.pattern and not re.fullmatch(self.pattern, value):
                raise ValueError(f"{self.label}: значение {value!r} не подходит по форме")
            return value
        try:
            value = int(raw) if self.kind == "int" else float(raw)
        except (TypeError, ValueError) as exc:
            expected = "целым числом" if self.kind == "int" else "числом"
            raise ValueError(f"{self.label}: значение должно быть {expected}, а не {raw!r}") from exc
        if self.low is not None and value < self.low:
            raise ValueError(f"{self.label}: значение {value} меньше допустимого {self.low}")
        if self.high is not None and value > self.high:
            raise ValueError(f"{self.label}: значение {value} больше допустимого {self.high}")
        return value


TURN_PARAMETERS: tuple[TurnParameter, ...] = (
    TurnParameter(
        "llm_temperature",
        "LLM_TEMPERATURE",
        "температура модели",
        "float",
        low=-1.0,
        high=2.0,
        note="−1 означает «не передавать параметр серверу вовсе»",
    ),
    TurnParameter("llm_timeout", "LLM_TIMEOUT", "таймаут модели, с", "int", low=5, high=3600),
    TurnParameter("top_k", "RAG_TOP_K", "фрагментов из отчётов", "int", low=1, high=50),
    TurnParameter(
        "min_score",
        "RAG_MIN_SCORE",
        "порог близости",
        "float",
        low=0.0,
        high=1.0,
        note="ниже порога находка считается непокрытой корпусом и включает веб-поиск",
    ),
    TurnParameter(
        "report_budget_chars",
        "REPORT_BUDGET_CHARS",
        "бюджет отчётов, знаков",
        "int",
        low=500,
        high=100_000,
    ),
    TurnParameter(
        "web_budget_chars", "WEB_BUDGET_CHARS", "бюджет веба, знаков", "int", low=0, high=100_000
    ),
    TurnParameter(
        "fragment_cap_chars",
        "FRAGMENT_CAP_CHARS",
        "потолок одного фрагмента, знаков",
        "int",
        low=200,
        high=50_000,
    ),
    TurnParameter(
        "history_budget_chars",
        "HISTORY_BUDGET_CHARS",
        "бюджет истории, знаков",
        "int",
        low=0,
        high=100_000,
    ),
    TurnParameter(
        "history_turn_cap_chars",
        "HISTORY_TURN_CAP_CHARS",
        "потолок одного хода истории, знаков",
        "int",
        low=100,
        high=50_000,
    ),
    TurnParameter("web_results", "WEB_RESULTS", "веб-результатов", "int", low=1, high=50),
    TurnParameter(
        "web_region",
        "WEB_REGION",
        "регион веб-поиска",
        "text",
        pattern=r"[a-z]{2}-[a-z]{2}",
        note="две буквы языка и две буквы страны, например ru-ru",
    ),
)

_BY_NAME = {parameter.field: parameter for parameter in TURN_PARAMETERS}


def turn_parameter(name: str) -> TurnParameter:
    if name not in _BY_NAME:
        raise KeyError(
            f"{name!r} не входит в число правимых из интерфейса параметров хода; "
            f"правимые: {', '.join(sorted(_BY_NAME))}"
        )
    return _BY_NAME[name]


def overrides_path() -> Path:
    """Файл, в котором живут правки из интерфейса.

    ★Отдельный файл, а не `.env`. В `.env` лежат секреты и комментарии, его
    ведёт тот, кто разворачивает систему, и переписывать его из интерфейса
    значило бы стирать чужую работу и рисковать ключами. Правки интерфейса —
    это состояние приложения, а не его конфигурация, и живут они там же, где
    остальное состояние: в `data/`.
    """
    return Path(_env("SETTINGS_OVERRIDES", str(ROOT / "data" / "settings-overrides.json")))


def read_overrides() -> dict:
    """Сохранённые правки. Битый файл — не повод падать при старте, но и не тишина."""
    path = overrides_path()
    if not path.is_file():
        return {}
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"⚠ {path}: файл правок не прочитан ({exc}); работаю по .env", flush=True)
        return {}
    if not isinstance(stored, dict):
        print(f"⚠ {path}: ожидался объект, а лежит {type(stored).__name__}; работаю по .env")
        return {}
    return stored


def _write_overrides(values: dict) -> None:
    path = overrides_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(values, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _apply(name: str, value) -> None:
    # Settings заморожен намеренно: случайная правка настройки посреди работы —
    # источник неповторимых отказов. Здесь правка не случайная: имя проверено по
    # закрытому списку, значение разобрано и введено в границы.
    object.__setattr__(settings, name, value)


def set_turn_parameter(name: str, raw) -> object:
    """Изменить параметр хода: проверить, применить в этом процессе и сохранить.

    Порядок именно такой. Сначала разбор — иначе на диск уедет мусор. Потом
    правка живого объекта — чтобы следующий же вопрос считался по новому
    значению. И только потом запись, потому что переживание перезапуска
    ПРОЦЕССА и есть смысл этой возможности: настройка, забытая при перезапуске,
    называется не настройкой, а прихотью текущей вкладки.
    """
    parameter = turn_parameter(name)
    value = parameter.parse(raw)
    # ★ПРАВКА, СОВПАДАЮЩАЯ С `.env`, — НЕ ПРАВКА. Сохранить её значило бы
    # показывать расхождение там, где его нет: «сейчас 5, а .env говорит 5».
    # Поймано приёмкой в браузере, случайно — и оказалось настоящим дефектом:
    # предупреждение, которое загорается без повода, обесценивает себя ровно
    # тогда, когда повод появится.
    if value == getattr(env_settings, parameter.field):
        return reset_turn_parameter(name)
    _apply(name, value)
    stored = read_overrides()
    stored[name] = value
    _write_overrides(stored)
    return value


def reset_turn_parameter(name: str) -> object:
    """Вернуть параметру то значение, которое даёт `.env`, и забыть правку."""
    parameter = turn_parameter(name)
    value = getattr(env_settings, parameter.field)
    _apply(name, value)
    stored = read_overrides()
    stored.pop(name, None)
    _write_overrides(stored)
    return value


def _load_saved_overrides() -> None:
    """Поднять сохранённые правки при старте процесса.

    ★Непригодная правка не применяется и НЕ МОЛЧИТ. Пропустить её тихо значило
    бы, что человек видит в панели одно значение, а система работает по
    другому, — ровно то расхождение, против которого вся эта ветка и сделана.
    """
    for name, raw in read_overrides().items():
        if name not in _BY_NAME:
            print(f"⚠ правка {name!r} пропущена: такого параметра хода нет", flush=True)
            continue
        try:
            _apply(name, _BY_NAME[name].parse(raw))
        except ValueError as exc:
            print(f"⚠ правка {name!r} пропущена: {exc}", flush=True)


_load_saved_overrides()
