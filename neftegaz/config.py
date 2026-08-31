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
    "SETTING_SPECS",
    "SettingSpec",
    "TURN_PARAMETERS",
    "check_settings",
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
    # Окно контекста подключённой модели, в токенах. 0 — «не знаем», и тогда
    # сверка бюджетов не проводится: выдумывать окно за пользователя хуже, чем
    # молчать.
    #
    # ★ЗАЧЕМ ЭТО ЗДЕСЬ. Бюджеты контекста заданы в ЗНАКАХ, а окно модели меряется
    # в ТОКЕНАХ, и переводной коэффициент зависит от языка: замер на нашем
    # материале дал 2.48 знака на токен по английским таблицам STEO и 2.02 по
    # русским ответам системы (tiktoken, cl100k_base). То есть сумма бюджетов в
    # 15 000 знаков — это примерно 6000–7400 токенов ТОЛЬКО контекста, без
    # задания и без ответа. При окне 8k этого впритык, а узнаётся такое отказом
    # сервера посреди работы: ровно так система однажды и выродилась в свалку
    # сырых фрагментов.
    llm_context_tokens: int = field(default_factory=lambda: _env_int("LLM_CONTEXT_TOKENS", 0))

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
    report_budget_chars: int = field(default_factory=lambda: _env_int("REPORT_BUDGET_CHARS", 7000))
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
    # Через сколько дней ряд цен считается устаревшим и расчёт говорит об этом
    # вслух. 0 выключает проверку.
    #
    # ★7 ДНЕЙ, А НЕ 1: биржа не торгует в выходные и праздники, поэтому разрыв в
    # два-четыре дня — нормальное состояние свежего ряда, и предупреждение на нём
    # звучало бы почти всегда. Неделя означает уже другое: скрипт обновления не
    # запускали. Дата последнего наблюдения печаталась и раньше (Р-074), но она
    # НЕ ОЦЕНИВАЛАСЬ — заметить, что число месячной давности, должен был сам
    # читатель, глядя на дату в строке отчёта.
    prices_stale_days: int = field(default_factory=lambda: _env_int("PRICES_STALE_DAYS", 7))

    # ── тексты, принадлежащие заказчику ───────────────────────────────────
    #
    # Роль, область экспертизы, требования к стилю и реплика отказа — предмет
    # редакционной политики заказчика: тон общения принадлежит ему. Пустое
    # значение означает «наш текст», и это умолчание, а не заглушка.
    #
    # ★Правила при отсутствии данных сюда НЕ входят и заменены быть не могут:
    # на них стои́т сверка цитат и весь раздел о проверяемости. Они
    # приклеиваются кодом к любому тексту роли — см. `neftegaz/agent/prompts.py`.
    system_prompt_file: str = field(default_factory=lambda: _env("SYSTEM_PROMPT_FILE", ""))
    out_of_scope_file: str = field(default_factory=lambda: _env("OUT_OF_SCOPE_FILE", ""))


settings = Settings()
# Снимок того, что говорят `.env` и умолчания, — БЕЗ правок из интерфейса.
# ★Нужен затем, чтобы правка из интерфейса не превращалась в тихое расхождение:
# человек, который открыл `.env` и прочёл `RAG_TOP_K=5`, обязан узнать в панели,
# что в работе сейчас другое число, а не гадать, почему система ведёт себя иначе,
# чем написано в её конфигурации.
env_settings = Settings()


# ── реестр настроек: что считается пригодным значением ─────────────────────
#
# ★ОДИН РЕЕСТР НА ДВЕ РАБОТЫ. Границы нужны дважды: когда значение приходит из
# интерфейса и когда оно приходит из `.env`. Две копии границ разошлись бы —
# и разошлись бы молча, потому что расхождение видно только на кривом значении,
# то есть в тот единственный день, когда проверка и нужна. Поэтому запись одна,
# а «правится ли из интерфейса» — её признак.
#
# ★СПИСОК ПРАВИМЫХ ЗАКРЫТЫЙ, И В ЭТОМ ВЕСЬ СМЫСЛ. Настройки системы делятся по
# тому, что они ломают: параметры ХОДА действуют со следующего вопроса и не
# трогают ничего, кроме него; параметры КОРПУСА (модель эмбеддингов, размер
# фрагмента, коллекция) делают уже собранный индекс несогласованным с
# настройкой. Первые можно править из интерфейса безопасно, вторые — нельзя, и
# разделение обязано быть не советом в документации, а списком в коде: то, чего
# здесь нет, из интерфейса не правится вовсе.
#
# ★РЕЕСТР ОБЯЗАН НАКРЫВАТЬ ВСЕ ПОЛЯ `Settings` — это проверяется тестом. Поле,
# у которого замкнутого множества значений нет (имя модели, путь, ключ), стоит
# в реестре как `free` С ПРИЧИНОЙ. Разница между «проверять нечего» и «проверить
# забыли» видна только тогда, когда первое написано вслух.


# Потолок для текстов, которые заказчик подставляет вместо наших. Взят с
# десятикратным запасом к нашим собственным (роль и стиль — 1014 знаков, реплика
# отказа — 394): запас на подробную редакционную политику есть, а файл, взятый
# по ошибке (скажем, выгрузка отчёта), отвергается. ★Текст роли едет в КАЖДЫЙ
# запрос к модели, поэтому его длина — вопрос не аккуратности, а бюджета
# контекста: он вытесняет отчёты, ради которых запрос и делается.
MAX_PROMPT_TEXT_CHARS = 10_000


@dataclass(frozen=True)
class SettingSpec:
    """Одна настройка: имя поля, имя в `.env` и то, что считается пригодным."""

    field: str
    env: str
    label: str
    kind: str  # "int" | "float" | "text" | "choice" | "bool" | "textfile" | "free"
    low: float | None = None
    high: float | None = None
    pattern: str | None = None
    choices: tuple[str, ...] = ()
    allow_empty: bool = False
    # Правится ли из интерфейса. Умолчание — «нет»: расширение списка правимых
    # обязано быть решением, а не следствием того, что поле кто-то добавил.
    editable: bool = False
    note: str = ""

    def parse(self, raw):
        """Разобрать значение или отказать с внятной причиной.

        ★Отказ громкий, а не тихий откат к прежнему. Опечатка, молча
        превращённая в умолчание, выглядит как исправная система, которая
        почему-то делает не то. Именно так `ELASTICITY_SOURCE=mesured`
        переключал расчёт с измеренной эластичности на литературную, меняя
        числа в ответе и не говоря ни слова.
        """
        if self.kind == "free":
            # ★Проверять нечего, и это сказано вслух. Проверка «строка непуста»
            # здесь была бы вечно зелёной: пустое значение переменной `_env`
            # заменяет умолчанием, то есть до поля не доходит вовсе. Проверка,
            # которая не может упасть, не лучше отсутствующей.
            return str(raw)
        if self.kind == "bool":
            if isinstance(raw, bool):
                return raw
            raise ValueError(f"{self.label}: значение должно быть да/нет, а не {raw!r}")
        if self.kind == "choice":
            value = str(raw).strip()
            if value not in self.choices:
                raise ValueError(
                    f"{self.label}: значение {value!r} не из списка; "
                    f"допустимо: {', '.join(self.choices)}"
                )
            return value
        if self.kind == "textfile":
            # Путь к файлу с текстом, который заказчик пишет вместо нашего.
            # Пусто — законное значение: оно означает «взять наш текст».
            path_text = str(raw).strip()
            if not path_text:
                return path_text
            candidate = Path(path_text)
            if not candidate.is_file():
                raise ValueError(f"{self.label}: файла {path_text!r} нет или это не файл")
            try:
                content = candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise ValueError(f"{self.label}: файл {path_text!r} не читается ({exc})") from exc
            if not content.strip():
                # ★Пустой файл — это отказ, а не «текста нет». Молчаливый
                # переход к умолчанию дал бы заказчику нашу роль вместо его
                # собственной, и он не узнал бы об этом никогда.
                raise ValueError(f"{self.label}: файл {path_text!r} пуст")
            if len(content) > MAX_PROMPT_TEXT_CHARS:
                raise ValueError(
                    f"{self.label}: в файле {path_text!r} {len(content)} знаков при потолке "
                    f"{MAX_PROMPT_TEXT_CHARS}; текст роли едет в КАЖДЫЙ запрос и вытеснил бы "
                    "из контекста сами отчёты"
                )
            return path_text
        if self.kind == "text":
            value = str(raw).strip()
            if not value:
                if self.allow_empty:
                    return value
                raise ValueError(f"{self.label}: значение не должно быть пустым")
            if self.pattern and not re.fullmatch(self.pattern, value):
                raise ValueError(
                    f"{self.label}: значение {value!r} не подходит по форме"
                    + (f" ({self.note})" if self.note else "")
                )
            return value
        try:
            value = int(raw) if self.kind == "int" else float(raw)
        except (TypeError, ValueError) as exc:
            expected = "целым числом" if self.kind == "int" else "числом"
            raise ValueError(
                f"{self.label}: значение должно быть {expected}, а не {raw!r}"
            ) from exc
        if self.low is not None and value < self.low:
            raise ValueError(f"{self.label}: значение {value} меньше допустимого {self.low}")
        if self.high is not None and value > self.high:
            raise ValueError(f"{self.label}: значение {value} больше допустимого {self.high}")
        return value


def _free(field_name: str, env: str, label: str, why: str) -> SettingSpec:
    """Настройка без замкнутого множества значений — с причиной, а не молчанием."""
    return SettingSpec(field_name, env, label, "free", note=why)


SETTING_SPECS: tuple[SettingSpec, ...] = (
    # ── параметры хода: правятся из интерфейса ─────────────────────────────
    SettingSpec(
        "llm_temperature",
        "LLM_TEMPERATURE",
        "температура модели",
        "float",
        low=-1.0,
        high=2.0,
        note="−1 означает «не передавать параметр серверу вовсе»",
        editable=True,
    ),
    SettingSpec(
        "llm_timeout", "LLM_TIMEOUT", "таймаут модели, с", "int", low=5, high=3600, editable=True
    ),
    SettingSpec(
        "llm_context_tokens",
        "LLM_CONTEXT_TOKENS",
        "окно контекста модели, токенов",
        "int",
        low=0,
        high=10_000_000,
        note="0 означает «окно неизвестно», и тогда бюджеты не сверяются с ним",
    ),
    SettingSpec(
        "top_k", "RAG_TOP_K", "фрагментов из отчётов", "int", low=1, high=50, editable=True
    ),
    SettingSpec(
        "min_score",
        "RAG_MIN_SCORE",
        "порог близости",
        "float",
        low=0.0,
        high=1.0,
        note="ниже порога находка считается непокрытой корпусом и включает веб-поиск",
        editable=True,
    ),
    SettingSpec(
        "report_budget_chars",
        "REPORT_BUDGET_CHARS",
        "бюджет отчётов, знаков",
        "int",
        low=500,
        high=100_000,
        editable=True,
    ),
    SettingSpec(
        "web_budget_chars",
        "WEB_BUDGET_CHARS",
        "бюджет веба, знаков",
        "int",
        low=0,
        high=100_000,
        editable=True,
    ),
    SettingSpec(
        "fragment_cap_chars",
        "FRAGMENT_CAP_CHARS",
        "потолок одного фрагмента, знаков",
        "int",
        low=200,
        high=50_000,
        editable=True,
    ),
    SettingSpec(
        "history_budget_chars",
        "HISTORY_BUDGET_CHARS",
        "бюджет истории, знаков",
        "int",
        low=0,
        high=100_000,
        editable=True,
    ),
    SettingSpec(
        "history_turn_cap_chars",
        "HISTORY_TURN_CAP_CHARS",
        "потолок одного хода истории, знаков",
        "int",
        low=100,
        high=50_000,
        editable=True,
    ),
    SettingSpec(
        "web_results", "WEB_RESULTS", "веб-результатов", "int", low=1, high=50, editable=True
    ),
    SettingSpec(
        "web_region",
        "WEB_REGION",
        "регион веб-поиска",
        "text",
        pattern=r"[a-z]{2}-[a-z]{2}",
        note="две буквы языка и две буквы страны, например ru-ru",
        editable=True,
    ),
    # ── параметры корпуса: из интерфейса не правятся, но проверяются ───────
    #
    # Правка любого из них делает уже собранный индекс несогласованным с
    # настройкой, поэтому менять их можно только вместе с пересборкой. Но
    # ПРОВЕРЯТЬ их надо ровно так же: опечатка здесь портит не один ход, а весь
    # корпус, и обходится дороже.
    SettingSpec(
        "chunk_size", "CHUNK_SIZE", "размер фрагмента, знаков", "int", low=100, high=20_000
    ),
    SettingSpec(
        "chunk_overlap",
        "CHUNK_OVERLAP",
        "перекрытие фрагментов, знаков",
        "int",
        low=0,
        high=10_000,
        note="обязано быть меньше размера фрагмента — см. проверку связей ниже",
    ),
    # ── откуда берётся эластичность ───────────────────────────────────────
    #
    # ★ЗАМКНУТОЕ МНОЖЕСТВО ИЗ ДВУХ ЗНАЧЕНИЙ, И ЭТО ТОТ САМЫЙ СЛУЧАЙ, РАДИ
    # КОТОРОГО ПРОВЕРКА ЗАВЕДЕНА. `ELASTICITY_SOURCE=mesured` до этой проверки
    # молча переключал расчёт на литературные числа: опечатка меняла цифры в
    # ответе и не говорила ни слова.
    SettingSpec(
        "elasticity_source",
        "ELASTICITY_SOURCE",
        "откуда берётся эластичность",
        "choice",
        choices=("measured", "literature"),
    ),
    # ── числа сценарного расчёта ──────────────────────────────────────────
    #
    # Эластичности обязаны быть ОТРИЦАТЕЛЬНЫМИ: рост цены снижает потребление.
    # Положительное значение перевернуло бы знак сценария — подорожание после
    # сокращения добычи стало бы удешевлением, и выглядело бы это как результат
    # расчёта, а не как опечатка.
    SettingSpec(
        "demand_elasticity_short",
        "DEMAND_ELASTICITY_SHORT",
        "эластичность на коротком горизонте",
        "float",
        low=-5.0,
        high=-0.01,
    ),
    SettingSpec(
        "demand_elasticity_long",
        "DEMAND_ELASTICITY_LONG",
        "эластичность на длинном горизонте",
        "float",
        low=-5.0,
        high=-0.01,
    ),
    SettingSpec(
        "demand_elasticity_band",
        "DEMAND_ELASTICITY_BAND",
        "полоса неопределённости эластичности",
        "float",
        low=-5.0,
        high=-0.01,
    ),
    SettingSpec(
        "elasticity_short_days",
        "ELASTICITY_SHORT_DAYS",
        "короткий горизонт, дней",
        "int",
        low=1,
        high=3650,
    ),
    SettingSpec(
        "elasticity_long_days",
        "ELASTICITY_LONG_DAYS",
        "длинный горизонт, дней",
        "int",
        low=1,
        high=36_500,
        note="обязан быть больше короткого — см. проверку связей ниже",
    ),
    SettingSpec(
        "global_supply_mb_d",
        "GLOBAL_SUPPLY_MB_D",
        "мировое предложение, млн барр./сут",
        "float",
        low=1.0,
        high=500.0,
    ),
    # ── память диалога ────────────────────────────────────────────────────
    #
    # ★Проверялось и раньше, но при ПЕРВОМ ИСПОЛЬЗОВАНИИ — то есть система
    # поднималась, показывала интерфейс и падала на первом же вопросе. Отказ
    # при старте называет ту же причину на несколько минут раньше и до того,
    # как человек успел задать вопрос.
    SettingSpec(
        "conversation_memory",
        "CONVERSATION_MEMORY",
        "память диалога",
        "choice",
        choices=("sqlite", "memory", "off"),
    ),
    SettingSpec("link_threads", "LINK_THREADS", "подключение разговоров", "bool"),
    # ── адреса служб ──────────────────────────────────────────────────────
    SettingSpec(
        "llm_base_url",
        "OPENAI_BASE_URL",
        "адрес модели",
        "text",
        pattern=r"https?://\S+",
        note="адрес вида http://хост:порт/v1",
    ),
    SettingSpec(
        "qdrant_url",
        "QDRANT_URL",
        "адрес Qdrant",
        "text",
        pattern=r"https?://\S+",
        allow_empty=True,
        note="пусто означает встроенный режим и файл на диске",
    ),
    # ── свободная форма: проверять нечего, и это сказано вслух ─────────────
    _free("llm_api_key", "OPENAI_API_KEY", "ключ модели", "ключ бывает любым"),
    _free("llm_model", "LLM_MODEL", "имя модели", "имя задаёт сервер модели, а не мы"),
    _free(
        "embedding_model",
        "EMBEDDING_MODEL",
        "модель эмбеддингов",
        "имя из каталога fastembed; список меняется в библиотеке, а не у нас",
    ),
    _free("qdrant_path", "QDRANT_PATH", "каталог Qdrant", "каталог создаётся при первом запуске"),
    _free("collection", "QDRANT_COLLECTION", "имя коллекции", "имя бывает любым"),
    # ★Проверяется по списку движков САМОЙ библиотеки и отвечает статусом, а не
    # падением (см. neftegaz/tools/web.py). Дублировать список здесь значило бы
    # завести копию, которая отстанет от библиотеки и начнёт врать; ронять же
    # старт из-за веб-поиска нельзя — система обязана работать по отчётам и без
    # сети.
    _free("web_backend", "WEB_BACKEND", "сервис веб-поиска", "сверяется со списком библиотеки"),
    _free("checkpoint_db", "CHECKPOINT_DB", "файл разговоров", "файл создаётся при первом запуске"),
    _free("reports_dir", "REPORTS_DIR", "каталог отчётов", "каталог задаёт развёртывание"),
    _free("prices_csv", "PRICES_CSV", "файл ряда цен", "путь задаёт развёртывание"),
    SettingSpec(
        "prices_stale_days",
        "PRICES_STALE_DAYS",
        "срок свежести ряда цен, дней",
        "int",
        low=0,
        high=3650,
        note="0 выключает предупреждение об устаревшем ряде",
    ),
    SettingSpec(
        "system_prompt_file",
        "SYSTEM_PROMPT_FILE",
        "файл с текстом роли и стиля",
        "textfile",
        note="пусто означает наш текст; правила при отсутствии данных не заменяются",
    ),
    SettingSpec(
        "out_of_scope_file",
        "OUT_OF_SCOPE_FILE",
        "файл с репликой отказа вне компетенции",
        "textfile",
        note="пусто означает нашу реплику",
    ),
)

# Параметры хода — ПОДМНОЖЕСТВО реестра, а не отдельный список: границы, по
# которым интерфейс проверяет введённое, обязаны быть теми же, по которым старт
# проверяет `.env`.
TURN_PARAMETERS: tuple[SettingSpec, ...] = tuple(spec for spec in SETTING_SPECS if spec.editable)

_BY_NAME = {parameter.field: parameter for parameter in TURN_PARAMETERS}


def check_settings(values: Settings) -> list[str]:
    """Перечислить всё непригодное в настройках. Пустой список — всё в порядке.

    ★ВСЕ находки разом, а не первая. Разворачивающий систему чинит `.env` за
    один заход, а не запускает её пять раз, узнавая по одной опечатке.

    ★Сообщение называет ИМЯ ПЕРЕМЕННОЙ, а не имя поля. Человек правит `.env`,
    в котором нет ни `min_score`, ни `conversation_memory`; сообщение, которое
    нельзя перенести в правку одним движением, — это полсообщения.
    """
    problems: list[str] = []
    for spec in SETTING_SPECS:
        try:
            spec.parse(getattr(values, spec.field))
        except ValueError as exc:
            problems.append(f"{spec.env} — {exc}")

    # ── связи между полями ────────────────────────────────────────────────
    #
    # ★Каждое значение по отдельности пригодно, а пара — нет. Такую пару не
    # поймает никакая проверка одного поля, а ведёт она себя тихо: перекрытие
    # больше фрагмента даёт бесконечное нарезание, длинный горизонт меньше
    # короткого — интерполяцию задом наперёд.
    if values.chunk_overlap >= values.chunk_size:
        problems.append(
            f"CHUNK_OVERLAP: перекрытие {values.chunk_overlap} не меньше размера "
            f"фрагмента CHUNK_SIZE={values.chunk_size}; нарезка не сдвигалась бы вперёд"
        )
    if values.elasticity_long_days <= values.elasticity_short_days:
        problems.append(
            f"ELASTICITY_LONG_DAYS: длинный горизонт {values.elasticity_long_days} не больше "
            f"короткого ELASTICITY_SHORT_DAYS={values.elasticity_short_days}; "
            "интерполяция эластичности встала бы задом наперёд"
        )
    return problems


# ★ЧИСЛА ПЕРЕВОДА ВЗЯТЫ ИЗ ЗАМЕРА, А НЕ ИЗ ГОЛОВЫ. На нашем материале
# (tiktoken, cl100k_base): 2.48 знака на токен по английским таблицам STEO,
# 2.02 по русским ответам системы. Берётся ХУДШИЙ конец: недооценка числа
# токенов — это ровно тот отказ, который мы предупреждаем.
# ⚠Оценка грубая по устройству: считает её токенизатор семейства OpenAI, а
# модель настраивается и может быть любой (Р-004). Поэтому результат — ПОВОД
# ПОСМОТРЕТЬ, а не отказ: ронять старт на чужом токенизаторе нельзя.
CHARS_PER_TOKEN = 2.0
# Запас на само задание и на ответ модели: контекстом окно не исчерпывается.
# Ответы демонстрационных сценариев — 1500–2500 знаков, то есть около тысячи
# токенов; задание с ролью и правилами — примерно столько же.
TOKENS_RESERVED_FOR_PROMPT_AND_ANSWER = 1536


def context_budget_warning(values: Settings) -> str | None:
    """Предупредить, если бюджеты контекста не помещаются в окно модели.

    Возвращает текст предупреждения или ``None``. ★Именно предупреждение, а не
    отказ: окно объявляет человек, перевод знаков в токены оценочный, и падать
    при старте из-за грубой оценки значило бы менять один тихий отказ на другой,
    громкий и часто ложный.
    """
    if values.llm_context_tokens <= 0:
        return None
    chars = values.report_budget_chars + values.web_budget_chars + values.history_budget_chars
    needed = int(chars / CHARS_PER_TOKEN) + TOKENS_RESERVED_FOR_PROMPT_AND_ANSWER
    if needed <= values.llm_context_tokens:
        return None
    return (
        f"⚠ бюджеты контекста ({chars} знаков) при худшем измеренном отношении "
        f"{CHARS_PER_TOKEN} знака на токен дают около {needed} токенов вместе с заданием и "
        f"ответом, а LLM_CONTEXT_TOKENS={values.llm_context_tokens}. Сервер модели ответит "
        "отказом «превышена длина контекста», и ответ выродится в свалку фрагментов. "
        "Уменьшите REPORT_BUDGET_CHARS / WEB_BUDGET_CHARS / HISTORY_BUDGET_CHARS "
        "или укажите настоящее окно модели."
    )


def _check_at_start(values: Settings) -> None:
    """Отказаться работать на непригодной настройке — при старте, а не потом.

    ★ЦЕНА РЕШЕНИЯ НАЗВАНА: запуск с наполовину заполненным `.env` перестаёт
    доходить до первого вопроса. Это и есть выигрыш: прежде такой запуск
    доходил, показывал интерфейс и вёл себя не так, как написано в его же
    конфигурации.
    """
    problems = check_settings(values)
    if not problems:
        return
    raise ValueError(
        "Настройки непригодны, система не запущена. Что править в .env:\n  - "
        + "\n  - ".join(problems)
    )


def turn_parameter(name: str) -> SettingSpec:
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

# ★ПРОВЕРКА ЗДЕСЬ, В КОНЦЕ МОДУЛЯ, А НЕ В СБОРКЕ `Settings`. Проверяется то, по
# чему система будет РАБОТАТЬ: `.env` плюс поднятые правки интерфейса, а не
# промежуточное состояние. И происходит это при импорте конфигурации, то есть
# раньше всего остального: ни один вопрос не будет отвечен по настройке, о
# которой мы знаем, что она непригодна.
_check_at_start(settings)

# Предупреждение (а не отказ) о бюджетах, не помещающихся в объявленное окно.
_warning = context_budget_warning(settings)
if _warning:
    print(_warning, flush=True)
